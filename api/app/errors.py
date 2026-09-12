"""Application error types and the handlers that turn them into responses.

Every failure that reaches a client carries three things: an HTTP status, a
stable machine-readable ``code``, and a message safe to show a user. Anything
raised that is not an AppError is treated as a bug: logged with a traceback,
reported as a generic 500.

The envelope, without exception:

    {"error": {"code": ..., "message": ..., "requestId": ..., "details": ...}}
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging_setup import get_logger, request_id_var

logger = get_logger(__name__)


class AppError(Exception):
    status: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code
        self.details = details

    @property
    def expected(self) -> bool:
        """4xx is the caller's problem and logs at warn; 5xx is ours."""
        return self.status < 500


class ValidationError(AppError):
    status = 400
    code = "validation_error"


class NotFoundError(AppError):
    status = 404
    code = "not_found"


class ConflictError(AppError):
    status = 409
    code = "conflict"


class UnprocessableError(AppError):
    """The request was fine but the system cannot serve it (e.g. the model declined)."""

    status = 422
    code = "unprocessable"


class UpstreamError(AppError):
    """A dependency we do not control failed: the fetched page, OpenAI, Anthropic."""

    status = 502
    code = "upstream_error"


def _envelope(code: str, message: str, details: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message, "requestId": request_id_var.get()}
    if details is not None:
        error["details"] = details
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        if exc.expected:
            logger.warning("request failed", extra={"code": exc.code, "status": exc.status, "reason": exc.message})
        else:
            logger.error(
                "request failed",
                extra={"code": exc.code, "status": exc.status, "reason": exc.message},
                exc_info=exc,
            )
        return JSONResponse(status_code=exc.status, content=_envelope(exc.code, exc.message, exc.details))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # A body that is not JSON at all is a different failure from a body whose
        # fields are wrong, and deserves its own code and message.
        if any(error["type"] == "json_invalid" for error in exc.errors()):
            logger.warning("malformed json body")
            return JSONResponse(
                status_code=400,
                content=_envelope("malformed_json", "Request body is not valid JSON."),
            )

        # Reshape FastAPI's errors into our own per-field details, so a client
        # sees one validation format across the whole API.
        issues = [
            {
                # loc is ("body", "content") / ("query", "limit"); drop the source segment.
                "field": ".".join(str(part) for part in error["loc"][1:]) or "(root)",
                "message": error["msg"],
                "code": error["type"],
            }
            for error in exc.errors()
        ]
        summary = "; ".join(f"{issue['field']} {issue['message']}" for issue in issues)
        logger.warning("request rejected", extra={"code": "validation_error", "issues": issues})
        return JSONResponse(
            status_code=400,
            content=_envelope("validation_error", f"Invalid request: {summary}", {"issues": issues}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        if exc.status_code == 404:
            return JSONResponse(
                status_code=404,
                content=_envelope("route_not_found", f"No route for {request.method} {request.url.path}"),
            )
        if exc.status_code == 405:
            return JSONResponse(
                status_code=405,
                content=_envelope("method_not_allowed", f"{request.method} is not allowed on {request.url.path}"),
            )
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope("http_error", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Detail stays in the logs, correlated by request id; the client gets nothing.
        logger.error("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=500,
            content=_envelope("internal_error", "Something went wrong on our side."),
        )
