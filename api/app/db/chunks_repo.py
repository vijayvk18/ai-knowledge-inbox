"""Chunk persistence, including the vector <-> BLOB conversion.

Vectors are stored as raw float32 little-endian bytes. A 1536-dim OpenAI vector
is 6 KB that way versus ~30 KB as JSON, and reading it back is a numpy
``frombuffer`` view rather than a parse.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from .database import get_connection, transaction


@dataclass(slots=True)
class Chunk:
    id: str
    item_id: str
    chunk_index: int
    content: str
    char_start: int
    char_end: int
    embedding: np.ndarray
    embedding_provider: str
    embedding_model: str
    dimensions: int
    # Joined from the parent item, for display and citation labels.
    item_title: str | None = None
    item_source_type: str | None = None
    item_url: str | None = None


def _to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        item_id=row["item_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        char_start=row["char_start"],
        char_end=row["char_end"],
        embedding=_from_blob(row["embedding"]),
        embedding_provider=row["embedding_provider"],
        embedding_model=row["embedding_model"],
        dimensions=row["dimensions"],
        item_title=row["item_title"] if "item_title" in row.keys() else None,
        item_source_type=row["item_source_type"] if "item_source_type" in row.keys() else None,
        item_url=row["item_url"] if "item_url" in row.keys() else None,
    )


def replace_for_item(
    item_id: str,
    chunks: list[dict],
    *,
    provider: str,
    model: str,
) -> list[Chunk]:
    """Replace every chunk for an item in one transaction.

    Re-ingesting is therefore idempotent: no orphans, no half-written chunk sets.
    """
    now = datetime.now(timezone.utc).isoformat()
    connection = get_connection()

    rows = [
        (
            f"chk_{uuid.uuid4()}",
            item_id,
            index,
            chunk["content"],
            chunk["char_start"],
            chunk["char_end"],
            _to_blob(chunk["embedding"]),
            provider,
            model,
            int(len(chunk["embedding"])),
            now,
        )
        for index, chunk in enumerate(chunks)
    ]

    with transaction():
        connection.execute("DELETE FROM chunks WHERE item_id = ?", (item_id,))
        connection.executemany(
            """INSERT INTO chunks (id, item_id, chunk_index, content, char_start, char_end,
                                   embedding, embedding_provider, embedding_model, dimensions, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        connection.commit()

    return [
        Chunk(
            id=row[0],
            item_id=item_id,
            chunk_index=row[2],
            content=row[3],
            char_start=row[4],
            char_end=row[5],
            embedding=_from_blob(row[6]),
            embedding_provider=provider,
            embedding_model=model,
            dimensions=row[9],
        )
        for row in rows
    ]


def list_all_with_item(*, provider: str, model: str) -> list[Chunk]:
    """Every fresh chunk joined to its parent's display fields - hydrates the index on boot.

    Filtered to the embedding that is live right now. Vectors written by a
    different provider *or a different version of the same one* are not
    comparable, and loading them would silently score every query against
    garbage - see find_stale_item_ids.
    """
    rows = get_connection().execute(
        """SELECT c.*, i.title AS item_title, i.source_type AS item_source_type, i.url AS item_url
             FROM chunks c
             JOIN items i ON i.id = c.item_id
            WHERE i.status = 'ready'
              AND c.embedding_provider = ?
              AND c.embedding_model = ?""",
        (provider, model),
    ).fetchall()
    return [_to_chunk(row) for row in rows]


def find_stale_item_ids(*, provider: str, model: str) -> list[str]:
    """Items whose vectors were written by a different embedding than the live one.

    Dimensions alone cannot detect this: changing a tokenizer changes what each
    dimension *means* while the vector stays the same width. The (provider,
    model) pair stored on every chunk is the honest identity, which is why the
    local provider's model string carries a version.
    """
    rows = get_connection().execute(
        """SELECT DISTINCT c.item_id
             FROM chunks c
             JOIN items i ON i.id = c.item_id
            WHERE i.status = 'ready'
              AND (c.embedding_provider != ? OR c.embedding_model != ?)""",
        (provider, model),
    ).fetchall()
    return [row["item_id"] for row in rows]


def count_all() -> int:
    return get_connection().execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]


def delete_for_item(item_id: str) -> int:
    connection = get_connection()
    with transaction():
        cursor = connection.execute("DELETE FROM chunks WHERE item_id = ?", (item_id,))
        connection.commit()
        return cursor.rowcount
