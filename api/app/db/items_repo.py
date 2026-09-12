"""Item persistence. SQL lives here and nowhere else.

Callers deal in ``Item`` dataclasses with camelCase-free field names; the
mapping to wire shapes happens in app/presenters.py.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .database import get_connection, transaction


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Item:
    id: str
    source_type: str
    title: str | None
    url: str | None
    raw_content: str | None
    status: str
    error: dict[str, str] | None
    char_count: int
    chunk_count: int
    created_at: str
    updated_at: str
    processed_at: str | None


@dataclass(slots=True)
class Cursor:
    created_at: str
    id: str


def _to_item(row: sqlite3.Row | None) -> Item | None:
    if row is None:
        return None
    return Item(
        id=row["id"],
        source_type=row["source_type"],
        title=row["title"],
        url=row["url"],
        raw_content=row["raw_content"],
        status=row["status"],
        error={"code": row["error_code"], "message": row["error_message"]} if row["error_code"] else None,
        char_count=row["char_count"],
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        processed_at=row["processed_at"],
    )


def create(
    *,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    raw_content: str | None = None,
    status: str = "pending",
) -> Item:
    item_id = f"itm_{uuid.uuid4()}"
    now = _now()
    connection = get_connection()

    with transaction():
        connection.execute(
            """INSERT INTO items (id, source_type, title, url, raw_content, status, char_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (item_id, source_type, title, url, raw_content, status, len(raw_content or ""), now, now),
        )
        connection.commit()

    found = find_by_id(item_id)
    assert found is not None  # just inserted
    return found


def find_by_id(item_id: str) -> Item | None:
    row = get_connection().execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
    return _to_item(row)


def list_items(
    *, limit: int = 50, status: str | None = None, cursor: Cursor | None = None
) -> tuple[list[Item], Cursor | None]:
    """Newest first, keyset-paginated on (created_at, id).

    Keyset rather than OFFSET so paging stays correct while new items are being
    ingested underneath the reader.
    """
    where: list[str] = []
    params: list[Any] = []

    if status:
        where.append("status = ?")
        params.append(status)
    if cursor:
        where.append("(created_at, id) < (?, ?)")
        params.extend([cursor.created_at, cursor.id])

    clause = f"WHERE {' AND '.join(where)}" if where else ""
    params.append(limit + 1)  # one extra row tells us whether more exist

    rows = get_connection().execute(
        f"SELECT * FROM items {clause} ORDER BY created_at DESC, id DESC LIMIT ?", params
    ).fetchall()

    has_more = len(rows) > limit
    page = rows[:limit] if has_more else rows
    next_cursor = Cursor(created_at=page[-1]["created_at"], id=page[-1]["id"]) if has_more and page else None

    return [item for item in (_to_item(row) for row in page) if item is not None], next_cursor


def count_by_status() -> dict[str, int]:
    rows = get_connection().execute("SELECT status, COUNT(*) AS count FROM items GROUP BY status").fetchall()
    return {row["status"]: row["count"] for row in rows}


def mark_processing(item_id: str) -> None:
    connection = get_connection()
    with transaction():
        connection.execute(
            "UPDATE items SET status = 'processing', updated_at = ? WHERE id = ?", (_now(), item_id)
        )
        connection.commit()


def mark_ready(item_id: str, *, title: str | None, url: str | None, raw_content: str, chunk_count: int) -> Item:
    now = _now()
    connection = get_connection()
    with transaction():
        connection.execute(
            """UPDATE items
                  SET status = 'ready',
                      title = COALESCE(?, title),
                      url = COALESCE(?, url),
                      raw_content = ?,
                      char_count = ?,
                      chunk_count = ?,
                      error_code = NULL,
                      error_message = NULL,
                      updated_at = ?,
                      processed_at = ?
                WHERE id = ?""",
            (title, url, raw_content, len(raw_content), chunk_count, now, now, item_id),
        )
        connection.commit()

    item = find_by_id(item_id)
    assert item is not None
    return item


def mark_failed(item_id: str, *, code: str, message: str) -> Item | None:
    now = _now()
    connection = get_connection()
    with transaction():
        connection.execute(
            """UPDATE items SET status = 'failed', error_code = ?, error_message = ?,
                      updated_at = ?, processed_at = ?
                WHERE id = ?""",
            (code, message[:500], now, now, item_id),
        )
        connection.commit()
    return find_by_id(item_id)


def find_unfinished() -> list[Item]:
    """Items left mid-flight by a crash or restart.

    The in-process queue lives in memory, so on boot these are handed back to it
    rather than stranded.
    """
    rows = get_connection().execute(
        "SELECT * FROM items WHERE status IN ('pending', 'processing') ORDER BY created_at ASC"
    ).fetchall()
    return [item for item in (_to_item(row) for row in rows) if item is not None]


def delete(item_id: str) -> bool:
    connection = get_connection()
    with transaction():
        cursor = connection.execute("DELETE FROM items WHERE id = ?", (item_id,))
        connection.commit()
        return cursor.rowcount > 0
