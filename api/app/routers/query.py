"""POST /api/query  - ask a question over saved content.
GET  /api/search - retrieval only, no LLM. A debugging surface.

/query answers with 200 even when it cannot answer (``answered: false``). An
empty inbox and an irrelevant question are ordinary outcomes for a knowledge
inbox, not HTTP errors, and the UI renders them as responses.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from ..config import settings
from ..logging_setup import get_logger
from ..rag.answer_service import answer_question
from ..rag.retriever import retrieve
from ..schemas import QueryRequest, QueryResponse, SearchResponse

router = APIRouter(tags=["query"])
logger = get_logger(__name__)


@router.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest) -> dict:
    logger.info("query received", extra={"questionChars": len(payload.question), "topK": payload.top_k})

    result = await answer_question(question=payload.question, top_k=payload.top_k)
    return {"question": payload.question, **result}


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(min_length=1),
    topK: int = Query(default=settings.retrieval_top_k, ge=1, le=settings.retrieval_max_top_k),
    # Override the relevance floor. minScore=0 shows what scored just below it,
    # which is the first thing to check when a query returns nothing.
    minScore: float | None = Query(default=None, ge=0.0, le=1.0),
) -> dict:
    passages, stats = await retrieve(question=q.strip(), top_k=topK, min_score=minScore)

    return {
        "question": q.strip(),
        "results": [
            {
                "itemId": passage["item_id"],
                "chunkId": passage["chunk_id"],
                "chunkIndex": passage["chunk_index"],
                "score": passage["score"],
                "title": passage["title"],
                "url": passage["url"],
                "sourceType": passage["source_type"],
                "snippet": passage["content"],
            }
            for passage in passages
        ],
        "stats": stats,
    }
