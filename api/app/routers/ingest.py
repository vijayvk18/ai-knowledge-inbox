"""POST /api/ingest - save a note or a URL.

Returns 202 Accepted, not 201: the row exists, but the content has not been
fetched, chunked, or embedded yet. The response carries the item in ``pending``
status and a Location header, and the client polls /api/items for the
transition to ``ready`` or ``failed``.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from ..ingest.service import get_ingestion_service
from ..presenters import to_item_summary
from ..schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse)
async def ingest(payload: IngestRequest, response: Response) -> dict:
    service = get_ingestion_service()

    if payload.type == "note":
        item = service.submit(source_type="note", content=payload.content, title=payload.title)
    else:
        item = service.submit(source_type="url", url=payload.url, title=payload.title)

    response.headers["Location"] = f"/api/items/{item.id}"
    return {
        "item": to_item_summary(item),
        "message": "Accepted. The item is being fetched and indexed.",
    }
