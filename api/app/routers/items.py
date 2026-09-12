"""Item listing and lifecycle.

    GET    /api/items             newest-first page + status counts
    GET    /api/items/{id}        full stored content
    POST   /api/items/{id}/retry  re-run a failed ingestion
    DELETE /api/items/{id}        remove the item, its chunks, and its vectors
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query, Response, status

from ..db import chunks_repo, items_repo
from ..db.items_repo import Item
from ..errors import ConflictError, NotFoundError, ValidationError
from ..ingest.service import get_ingestion_service
from ..logging_setup import get_logger
from ..presenters import decode_cursor, encode_cursor, to_item_detail, to_item_summary
from ..rag.vector_store import vector_store
from ..schemas import IngestResponse, ItemListResponse, ItemResponse

router = APIRouter(tags=["items"])
logger = get_logger(__name__)


def _require_item(item_id: str) -> Item:
    if not item_id.startswith("itm_"):
        raise ValidationError("Not a valid item id.", details={"id": item_id})
    item = items_repo.find_by_id(item_id)
    if item is None:
        raise NotFoundError(f"No item with id {item_id}")
    return item


@router.get("/items", response_model=ItemListResponse)
async def list_items(
    limit: int = Query(default=50, ge=1, le=100),
    status_filter: Literal["pending", "processing", "ready", "failed"] | None = Query(
        default=None, alias="status"
    ),
    cursor: str | None = Query(default=None),
) -> dict:
    decoded = decode_cursor(cursor)
    if decoded is False:
        raise ValidationError("The cursor is not valid. Start from the first page.")

    items, next_cursor = items_repo.list_items(limit=limit, status=status_filter, cursor=decoded)

    return {
        "items": [to_item_summary(item) for item in items],
        "nextCursor": encode_cursor(next_cursor),
        # Cheap enough to include on every page, and it drives the UI's status
        # filter without a second round trip.
        "counts": items_repo.count_by_status(),
    }


@router.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: str) -> dict:
    return {"item": to_item_detail(_require_item(item_id))}


@router.post("/items/{item_id}/retry", status_code=status.HTTP_202_ACCEPTED, response_model=IngestResponse)
async def retry_item(item_id: str) -> dict:
    item = _require_item(item_id)
    if item.status in ("pending", "processing"):
        raise ConflictError("That item is already being processed.")

    get_ingestion_service().enqueue(item.id)
    logger.info("item requeued", extra={"itemId": item.id, "previousStatus": item.status})

    summary = to_item_summary(item)
    summary["status"] = "pending"
    summary["error"] = None
    return {"item": summary, "message": "Accepted. The item is being re-indexed."}


@router.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_item(item_id: str) -> Response:
    item = _require_item(item_id)

    # Chunks go with the item via ON DELETE CASCADE; the in-memory index is
    # derived state and has to be told separately.
    chunks_repo.delete_for_item(item.id)
    items_repo.delete(item.id)
    vector_store.remove_item(item.id)

    logger.info("item deleted", extra={"itemId": item.id})
    return Response(status_code=status.HTTP_204_NO_CONTENT)
