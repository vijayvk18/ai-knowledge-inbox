"""GET /api/health - is the process up, and what is it configured with?

Deliberately more than {"ok": true}. Most confusing behaviour in this app
("why is search bad", "why is the answer just a quote") is explained by which
providers are live and how much is indexed, so the health check says so. No
secrets: provider and model names only.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from ..db import items_repo
from ..ingest.service import get_ingestion_service
from ..rag.answer_service import get_answer_provider
from ..rag.embeddings import get_embedding_provider
from ..rag.vector_store import vector_store
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])

_STARTED_AT = time.time()


@router.get("/health", response_model=HealthResponse)
async def health() -> dict:
    embeddings = get_embedding_provider()
    answers = get_answer_provider()

    return {
        "status": "ok",
        "uptimeSeconds": round(time.time() - _STARTED_AT),
        "providers": {
            "embeddings": {
                "provider": embeddings.name,
                "model": embeddings.model,
                "dimensions": embeddings.dimensions,
            },
            "answers": {"provider": answers.name, "model": answers.model},
        },
        "index": {
            "items": vector_store.item_count(),
            "chunks": vector_store.size(),
            "dimensions": vector_store.dimensions(),
        },
        "items": items_repo.count_by_status(),
        "ingestion": get_ingestion_service().stats(),
    }
