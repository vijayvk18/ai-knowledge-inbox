"""SQLite connection and schema bootstrap.

One process-wide connection guarded by a lock. Two notes on why that is the
right shape here rather than a pool:

* SQLite reads are microseconds, so there is no event-loop win from pushing
  them to a thread pool - the slow work in this app is network I/O (fetching
  pages, embedding, generating), which *is* async and does yield.
* ``check_same_thread=False`` plus an explicit lock keeps the ingestion worker
  and request handlers from interleaving writes. WAL lets readers proceed while
  a write is in flight.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from ..config import settings
from ..logging_setup import get_logger

logger = get_logger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_connection: sqlite3.Connection | None = None
_lock = threading.RLock()


def get_connection() -> sqlite3.Connection:
    global _connection
    if _connection is not None:
        return _connection

    with _lock:
        if _connection is not None:
            return _connection

        db_file = settings.db_file
        if db_file != ":memory:":
            Path(db_file).parent.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(db_file, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        connection.commit()

        _connection = connection
        logger.info("database ready", extra={"file": db_file})
        return connection


def transaction() -> threading.RLock:
    """Guard for multi-statement writes: ``with transaction(): ...``."""
    return _lock


def close_connection() -> None:
    global _connection
    with _lock:
        if _connection is not None:
            _connection.close()
            _connection = None
