"""In-memory vector index over the chunks in SQLite.

SQLite is the source of truth; this is a derived read model rebuilt on boot and
kept in step by the ingestion pipeline. Search is a brute-force scan - but a
vectorized one: all vectors live in a single ``(n, d)`` float32 matrix and a
query is one ``matrix @ query`` BLAS call.

Why brute force? For a single-user inbox it is both simpler and faster than an
ANN index. 10k chunks x 1536 dims is ~60 MB and one matrix-vector product -
single-digit milliseconds under numpy, with exact recall, no index build, no
tuning, no extra dependency. The scan is linear, so the point where this stops
being the right answer is real and documented in the README (~10^5 chunks).

Vectors are unit-normalized on insert, so cosine similarity is a dot product.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from ..db import chunks_repo
from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class IndexedChunk:
    chunk_id: str
    item_id: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    item_title: str | None
    item_source_type: str | None
    item_url: str | None


@dataclass(slots=True)
class SearchHit:
    chunk_id: str
    item_id: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    item_title: str | None
    item_source_type: str | None
    item_url: str | None
    score: float


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0 or abs(norm - 1.0) < 1e-6:
        return vector
    return (vector / norm).astype(np.float32)


class VectorStore:
    def __init__(self) -> None:
        self._by_item: dict[str, list[tuple[IndexedChunk, np.ndarray]]] = {}
        self._lock = threading.RLock()
        # Rebuilt lazily: the matrix is a cache over _by_item, not a second truth.
        self._matrix: np.ndarray | None = None
        self._entries: list[IndexedChunk] = []
        self._dirty = True

    # ---------- maintenance ----------

    def hydrate(self, *, provider: str, model: str) -> "VectorStore":
        """Rebuild the index from SQLite, loading only vectors from the live embedding.

        Vectors written by a different provider - or a different version of the
        same one - are left out rather than scored against, because they are not
        in the same space. IngestionService.reindex_stale re-embeds them.
        """
        with self._lock:
            self._by_item.clear()
            for chunk in chunks_repo.list_all_with_item(provider=provider, model=model):
                entry = IndexedChunk(
                    chunk_id=chunk.id,
                    item_id=chunk.item_id,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    item_title=chunk.item_title,
                    item_source_type=chunk.item_source_type,
                    item_url=chunk.item_url,
                )
                self._by_item.setdefault(chunk.item_id, []).append((entry, _normalize(chunk.embedding)))
            self._dirty = True

        logger.info("vector index hydrated", extra={"items": self.item_count(), "chunks": self.size()})
        return self

    def set_item_chunks(
        self,
        item_id: str,
        chunks: list,
        *,
        title: str | None = None,
        source_type: str | None = None,
        url: str | None = None,
    ) -> None:
        """Replace every vector for one item. Mirrors chunks_repo.replace_for_item."""
        with self._lock:
            self._by_item[item_id] = [
                (
                    IndexedChunk(
                        chunk_id=chunk.id,
                        item_id=item_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        item_title=title,
                        item_source_type=source_type,
                        item_url=url,
                    ),
                    _normalize(chunk.embedding),
                )
                for chunk in chunks
            ]
            self._dirty = True

    def remove_item(self, item_id: str) -> None:
        with self._lock:
            if self._by_item.pop(item_id, None) is not None:
                self._dirty = True

    def clear(self) -> None:
        with self._lock:
            self._by_item.clear()
            self._dirty = True

    # ---------- introspection ----------

    def size(self) -> int:
        return sum(len(entries) for entries in self._by_item.values())

    def item_count(self) -> int:
        return len(self._by_item)

    def dimensions(self) -> int | None:
        for entries in self._by_item.values():
            for _, vector in entries:
                return int(vector.shape[0])
        return None

    # ---------- search ----------

    def _rebuild(self, dimension: int) -> None:
        """Pack vectors of the given dimension into one contiguous matrix.

        Vectors of another dimension are skipped rather than crashing: that is
        what an embedding-provider switch without a re-index looks like, and
        silently scoring them would return nonsense.
        """
        entries: list[IndexedChunk] = []
        vectors: list[np.ndarray] = []
        skipped = 0

        for item_entries in self._by_item.values():
            for entry, vector in item_entries:
                if vector.shape[0] != dimension:
                    skipped += 1
                    continue
                entries.append(entry)
                vectors.append(vector)

        self._entries = entries
        self._matrix = np.vstack(vectors).astype(np.float32) if vectors else None
        self._dirty = False

        if skipped:
            logger.warning(
                "skipped vectors with mismatched dimensions - re-ingest after changing embedding provider",
                extra={"skipped": skipped, "expected": dimension},
            )

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        max_per_item: int | None = None,
    ) -> tuple[list[SearchHit], int]:
        """Top-k nearest chunks by cosine similarity.

        ``max_per_item`` keeps a single long document from filling the whole
        context window: the scan is global, the cap is applied while collecting
        results, so a long article competes on quality rather than chunk count.
        """
        query = _normalize(query_vector)

        with self._lock:
            if self._dirty or (self._matrix is not None and self._matrix.shape[1] != query.shape[0]):
                self._rebuild(int(query.shape[0]))

            if self._matrix is None or not self._entries:
                return [], 0

            scores = self._matrix @ query
            above_floor = int(np.count_nonzero(scores >= min_score))
            if above_floor == 0:
                return [], 0

            # Partial sort: only the plausible candidates are fully ordered.
            candidate_count = min(len(scores), max(top_k * (max_per_item or 1), top_k) + 32)
            candidate_idx = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
            candidate_idx = candidate_idx[np.argsort(-scores[candidate_idx])]

            cap = max_per_item if max_per_item is not None else len(self._entries)
            per_item: dict[str, int] = {}
            hits: list[SearchHit] = []

            for index in candidate_idx:
                score = float(scores[index])
                if score < min_score:
                    break
                if len(hits) >= top_k:
                    break

                entry = self._entries[index]
                used = per_item.get(entry.item_id, 0)
                if used >= cap:
                    continue
                per_item[entry.item_id] = used + 1

                hits.append(
                    SearchHit(
                        chunk_id=entry.chunk_id,
                        item_id=entry.item_id,
                        chunk_index=entry.chunk_index,
                        content=entry.content,
                        char_start=entry.char_start,
                        char_end=entry.char_end,
                        item_title=entry.item_title,
                        item_source_type=entry.item_source_type,
                        item_url=entry.item_url,
                        score=score,
                    )
                )

            return hits, above_floor


# The process-wide index. Single-user app, single index.
vector_store = VectorStore()
