"""Structured logging.

One line per event, JSON in production so it can be shipped and queried, and a
short human form in development. Extra fields travel as a dict in `extra=`, and
a request id is bound per-request via a ContextVar so it lands on every line
without being threaded through every function signature.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from .config import settings

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

# Anything the stdlib puts on a LogRecord; everything else is ours.
_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


def _extra_fields(record: logging.LogRecord) -> dict[str, Any]:
    return {key: value for key, value in record.__dict__.items() if key not in _RESERVED}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["requestId"] = request_id
        payload.update(_extra_fields(record))
        if record.exc_info:
            payload["stack"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        fields = _extra_fields(record)
        request_id = request_id_var.get()
        if request_id:
            fields["requestId"] = request_id
        detail = f" {json.dumps(fields, default=str)}" if fields else ""
        stamp = datetime.now(timezone.utc).isoformat()
        line = f"{stamp} {record.levelname.ljust(5)} {record.getMessage()}{detail}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(PrettyFormatter() if settings.log_pretty else JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())

    # Uvicorn ships its own handlers; route them through ours so the output is
    # one consistent stream rather than two competing formats.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # Access lines are redundant: our middleware logs richer ones.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
