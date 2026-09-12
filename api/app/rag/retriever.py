"""Retrieval: question -> ranked, citation-numbered passages.

Kept separate from answer generation so the retrieval half can be tested,
tuned, and inspected on its own - it is where RAG quality is usually won or
lost, and GET /api/search exposes exactly this.
"""

from __future__ import annotations

import time
from typing import Any

from ..config import settings
from .embeddings import get_embedding_provider
from .vector_store import SearchHit, vector_store


def _source_label(hit: SearchHit) -> str:
    """A short human label for a source, used in the prompt and in the UI."""
    title = (hit.item_title or "").strip()
    if hit.item_source_type == "url":
        return f"{title} ({hit.item_url})" if title else (hit.item_url or "Saved page")
    return f"Note: {title}" if title else "Note"


def _apply_context_budget(hits: list[SearchHit], max_chars: int) -> list[tuple[SearchHit, str, bool]]:
    """Trim the passage list to the context budget.

    Passages arrive best-first, so truncating from the end drops the weakest
    evidence. A character budget is cheaper and more predictable than token
    counting, and it keeps one pathological chunk from blowing up prompt cost.
    """
    kept: list[tuple[SearchHit, str, bool]] = []
    used = 0

    for hit in hits:
        cost = len(hit.content)
        if used + cost > max_chars:
            # Always keep at least the top hit, truncated, rather than returning
            # nothing to answer from.
            if not kept:
                kept.append((hit, hit.content[:max_chars], True))
            break
        kept.append((hit, hit.content, False))
        used += cost

    return kept


async def retrieve(
    *, question: str, top_k: int | None = None, min_score: float | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    provider = get_embedding_provider()
    k = top_k if top_k is not None else settings.retrieval_top_k
    floor = min_score
    if floor is None:
        floor = settings.retrieval_min_score
    if floor is None:
        floor = getattr(provider, "default_min_score", 0.0)

    started = time.perf_counter()
    query_vector = await provider.embed_query(question)
    embed_ms = (time.perf_counter() - started) * 1000

    search_started = time.perf_counter()
    hits, above_floor = vector_store.search(
        query_vector, top_k=k, min_score=floor, max_per_item=settings.retrieval_max_per_item
    )
    search_ms = (time.perf_counter() - search_started) * 1000

    passages = [
        {
            "citation": index + 1,
            "chunk_id": hit.chunk_id,
            "item_id": hit.item_id,
            "chunk_index": hit.chunk_index,
            "source_type": hit.item_source_type,
            "title": hit.item_title,
            "url": hit.item_url,
            "source_label": _source_label(hit),
            "content": content,
            "char_start": hit.char_start,
            "char_end": hit.char_end,
            "truncated": truncated,
            "score": round(hit.score, 4),
        }
        for index, (hit, content, truncated) in enumerate(
            _apply_context_budget(hits, settings.retrieval_max_context_chars)
        )
    ]

    stats = {
        "indexedChunks": vector_store.size(),
        "candidatesAboveFloor": above_floor,
        "minScore": floor,
        "topK": k,
        "embeddingMs": round(embed_ms),
        "searchMs": round(search_ms),
        "embeddingProvider": provider.name,
        "embeddingModel": provider.model,
    }
    return passages, stats
