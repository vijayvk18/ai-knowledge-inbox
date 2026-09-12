"""Ingestion pipeline.

* ``submit()``  - validate, persist a ``pending`` row, enqueue, return immediately
* ``_process()`` - resolve content -> chunk -> embed -> persist -> index -> ``ready``

This is the async boundary of the app. POST /ingest returns as soon as the row
is written; fetching, chunking, and embedding happen on a pool of asyncio
worker tasks, because they take seconds and can fail for reasons the caller
cannot act on synchronously.

The queue is in-process on purpose - a single-user inbox needs no broker. The
cost is explicit: work queued in memory is lost if the process dies, which is
why item status lives in SQLite and unfinished rows are re-queued on boot.

Every failure path ends with the item marked ``failed`` and a message the user
can act on, because an item that silently disappears is the worst outcome in an
inbox.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from ..config import settings
from ..db import chunks_repo, items_repo
from ..errors import AppError, ValidationError
from ..logging_setup import get_logger
from ..rag.chunker import chunk_text
from ..rag.embeddings import get_embedding_provider
from ..rag.vector_store import vector_store
from .html_extract import extract_from_plain_text, extract_readable_text
from .url_fetcher import fetch_url, parse_and_validate_url

logger = get_logger(__name__)


@dataclass(slots=True, frozen=True)
class _Job:
    item_id: str
    # Re-embed from the text already in SQLite instead of re-fetching the URL.
    # Used when the embedding changed but the content did not.
    use_stored_content: bool = False


def _title_from_text(text: str) -> str | None:
    """First non-empty line, trimmed to something that fits in a list row."""
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            return f"{stripped[:77]}..." if len(stripped) > 80 else stripped
    return None


class IngestionService:
    def __init__(self, *, concurrency: int | None = None) -> None:
        self._concurrency = concurrency if concurrency is not None else settings.ingest_concurrency
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._workers: list[asyncio.Task] = []
        # Cheap idempotency: never run the same item twice at once.
        self._known: set[str] = set()

    # ---------- lifecycle ----------

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"ingest-worker-{index}")
            for index in range(self._concurrency)
        ]
        logger.info("ingestion workers started", extra={"concurrency": self._concurrency})

    async def stop(self) -> None:
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []

    async def _worker(self, index: int) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process(job)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a bug in _process, which owns failure reporting
                logger.error(
                    "job handler raised", extra={"itemId": job.item_id, "worker": index}, exc_info=exc
                )
            finally:
                self._known.discard(job.item_id)
                self._queue.task_done()

    # ---------- public API ----------

    def submit(self, *, source_type: str, content: str | None = None, url: str | None = None, title: str | None = None):
        if source_type == "note":
            assert content is not None
            item = items_repo.create(
                source_type="note",
                title=title or _title_from_text(content),
                raw_content=content,
            )
        else:
            assert url is not None
            item = items_repo.create(
                source_type="url",
                title=title,
                # Validate before writing a row, so a malformed URL is a 400 rather
                # than a failed item the user has to go clean up.
                url=parse_and_validate_url(url),
            )

        self.enqueue(item.id)
        logger.info("item queued", extra={"itemId": item.id, "sourceType": item.source_type})
        return item

    def enqueue(self, item_id: str, *, use_stored_content: bool = False) -> bool:
        if item_id in self._known:
            logger.debug("job already queued, skipping", extra={"itemId": item_id})
            return False
        self._known.add(item_id)
        self._queue.put_nowait(_Job(item_id=item_id, use_stored_content=use_stored_content))
        return True

    def recover_unfinished(self) -> int:
        """Hand back anything the last process left mid-flight.

        In-memory queues do not survive restarts; the status column does.
        """
        stranded = items_repo.find_unfinished()
        for item in stranded:
            self.enqueue(item.id)
        if stranded:
            logger.info("requeued unfinished items", extra={"count": len(stranded)})
        return len(stranded)

    def reindex_stale(self) -> int:
        """Re-embed items whose vectors predate the current embedding model.

        Changing the embedding does not change the saved text, so this re-chunks
        and re-embeds from SQLite - no refetch, no risk of a page having changed
        or gone away underneath the user. Without it, those items stay in the
        database scoring zero against every query, which looks exactly like
        "nothing matches" with no clue as to why.
        """
        provider = get_embedding_provider()
        stale = chunks_repo.find_stale_item_ids(provider=provider.name, model=provider.model)
        for item_id in stale:
            self.enqueue(item_id, use_stored_content=True)
        if stale:
            logger.warning(
                "re-indexing items embedded with a different model",
                extra={"count": len(stale), "provider": provider.name, "model": provider.model},
            )
        return len(stale)

    def stats(self) -> dict[str, int]:
        return {
            "queued": self._queue.qsize(),
            "inFlight": len(self._known) - self._queue.qsize(),
            "concurrency": self._concurrency,
        }

    async def wait_until_idle(self) -> None:
        """Resolves when the queue drains. Used by tests and graceful shutdown."""
        await self._queue.join()

    # ---------- pipeline ----------

    async def _resolve_content(self, item, *, use_stored_content: bool) -> tuple[str, str | None, str | None]:
        # Re-embedding existing text: the content is already right, only its
        # vector is stale. Skip the network entirely - no risk of the page having
        # changed or gone away since the user saved it.
        if use_stored_content and item.raw_content:
            return item.raw_content, item.title, item.url

        if item.source_type == "note":
            return item.raw_content or "", item.title or _title_from_text(item.raw_content or ""), None

        fetched = await fetch_url(item.url)
        extracted = (
            extract_readable_text(fetched.body)
            if "html" in fetched.content_type
            else extract_from_plain_text(fetched.body, fetched.final_url)
        )

        logger.info(
            "url fetched",
            extra={
                "url": fetched.final_url,
                "contentType": fetched.content_type.split(";")[0],
                "extractedChars": len(extracted.text),
                "truncated": fetched.truncated,
            },
        )
        return extracted.text, item.title or extracted.title or fetched.final_url, fetched.final_url

    async def _process(self, job: _Job) -> None:
        item_id = job.item_id
        item = items_repo.find_by_id(item_id)
        if item is None:
            logger.warning("item disappeared before processing", extra={"itemId": item_id})
            return
        if item.status == "ready" and not job.use_stored_content:
            logger.debug("item already processed, skipping", extra={"itemId": item_id})
            return

        started = time.perf_counter()
        items_repo.mark_processing(item_id)

        try:
            raw_text, title, final_url = await self._resolve_content(
                item, use_stored_content=job.use_stored_content
            )

            if not raw_text or not raw_text.strip():
                raise ValidationError(
                    "No readable text could be extracted from that page."
                    if item.source_type == "url"
                    else "That note has no text to index."
                )

            text, chunks = chunk_text(raw_text)
            if not chunks:
                raise ValidationError("That content produced no indexable chunks.")

            provider = get_embedding_provider()
            embed_started = time.perf_counter()
            vectors = await provider.embed_documents([chunk.content for chunk in chunks])
            embed_ms = round((time.perf_counter() - embed_started) * 1000)

            if len(vectors) != len(chunks):
                raise AppError(
                    "The embedding provider returned the wrong number of vectors.",
                    code="embedding_count_mismatch",
                    details={"expected": len(chunks), "received": len(vectors)},
                )

            payload = [
                {
                    "content": chunk.content,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "embedding": vector,
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]

            # SQLite first, then the in-memory index: the index is derived data,
            # and if the process dies between the two it is rebuilt on boot.
            stored = chunks_repo.replace_for_item(
                item_id, payload, provider=provider.name, model=provider.model
            )

            ready = items_repo.mark_ready(
                item_id, title=title, url=final_url, raw_content=text, chunk_count=len(stored)
            )

            vector_store.set_item_chunks(
                item_id,
                stored,
                title=ready.title,
                source_type=ready.source_type,
                url=ready.url,
            )

            logger.info(
                "item ingested",
                extra={
                    "itemId": item_id,
                    "sourceType": item.source_type,
                    "chars": len(text),
                    "chunks": len(stored),
                    "embedMs": embed_ms,
                    "totalMs": round((time.perf_counter() - started) * 1000),
                    "embeddingProvider": provider.name,
                    "reindex": job.use_stored_content,
                },
            )

        except AppError as error:
            if error.expected:
                logger.warning(
                    "ingestion failed", extra={"itemId": item_id, "code": error.code, "reason": error.message}
                )
                items_repo.mark_failed(item_id, code=error.code, message=error.message)
            else:
                logger.error("ingestion crashed", extra={"itemId": item_id, "code": error.code}, exc_info=error)
                items_repo.mark_failed(
                    item_id, code=error.code, message="Something went wrong while processing this item."
                )
        except Exception as error:
            logger.error("ingestion crashed", extra={"itemId": item_id}, exc_info=error)
            items_repo.mark_failed(
                item_id,
                code="internal_error",
                message="Something went wrong while processing this item.",
            )


_service: IngestionService | None = None


def get_ingestion_service() -> IngestionService:
    global _service
    if _service is None:
        _service = IngestionService()
    return _service


def reset_ingestion_service() -> None:
    """Test seam."""
    global _service
    _service = None
