"""Request id + access logging.

Every request gets an id (reusing an inbound x-request-id when a proxy set
one), bound to a ContextVar so it lands on every log line emitted while
handling that request, and echoed back as a header. The id also appears in
every error body, so a user-reported failure can be grepped out of the logs.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .logging_setup import get_logger, request_id_var

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            # The exception handlers build the response; log the timing here so a
            # crashed request still produces an access line.
            logger.error(
                "request failed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": 500,
                    "durationMs": round((time.perf_counter() - started) * 1000),
                },
            )
            request_id_var.reset(token)
            raise

        duration_ms = round((time.perf_counter() - started) * 1000)
        fields = {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "durationMs": duration_ms,
        }
        # 5xx is our problem and belongs at error level; 4xx is the caller's.
        if response.status_code >= 500:
            logger.error("request failed", extra=fields)
        elif response.status_code >= 400:
            logger.warning("request rejected", extra=fields)
        else:
            logger.info("request completed", extra=fields)

        response.headers["x-request-id"] = request_id
        request_id_var.reset(token)
        return response
