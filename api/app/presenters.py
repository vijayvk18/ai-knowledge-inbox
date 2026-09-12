"""Wire shapes.

The API is not a view of the database. Listing items must not ship every saved
article in full, and cursors must be opaque, so the row -> response mapping
lives here rather than being improvised in each router.
"""

from __future__ import annotations

import base64
import json

from .db.items_repo import Cursor, Item

PREVIEW_CHARS = 240


def encode_cursor(cursor: Cursor | None) -> str | None:
    if cursor is None:
        return None
    raw = json.dumps({"createdAt": cursor.created_at, "id": cursor.id}).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(encoded: str | None) -> Cursor | None | bool:
    """Returns a Cursor, None when absent, or False when malformed.

    False rather than an exception so the router can turn it into a 400 with a
    message about starting from the first page.
    """
    if not encoded:
        return None
    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        parsed = json.loads(base64.urlsafe_b64decode(padded))
        if isinstance(parsed.get("createdAt"), str) and isinstance(parsed.get("id"), str):
            return Cursor(created_at=parsed["createdAt"], id=parsed["id"])
    except (ValueError, TypeError):
        pass
    return False


def to_item_summary(item: Item) -> dict:
    """Summary form, for lists. Carries a preview, never the full body."""
    preview = None
    if item.raw_content:
        preview = item.raw_content[:PREVIEW_CHARS]
        if len(item.raw_content) > PREVIEW_CHARS:
            preview += "..."

    return {
        "id": item.id,
        "sourceType": item.source_type,
        "title": item.title,
        "url": item.url,
        "status": item.status,
        "error": item.error,
        "charCount": item.char_count,
        "chunkCount": item.chunk_count,
        "preview": preview,
        "createdAt": item.created_at,
        "updatedAt": item.updated_at,
        "processedAt": item.processed_at,
    }


def to_item_detail(item: Item) -> dict:
    """Detail form, for GET /items/{id}. Adds the stored text."""
    return {**to_item_summary(item), "content": item.raw_content}
