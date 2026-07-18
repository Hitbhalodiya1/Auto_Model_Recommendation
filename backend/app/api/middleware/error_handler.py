"""
Global error handler middleware.
Translates domain exceptions into standardized HTTP error responses.
Stack traces are never exposed in production.
"""

import uuid
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.exceptions.domain_exceptions import (
    AutoRecError,
    FileTooLargeError,
    InvalidFileTypeError,
    InvalidStateError,
    MLError,
    NotFoundError,
    StorageError,
    ValidationError,
)

logger = get_logger(__name__)


def _error_response(
    request_id: str,
    status_code: int,
    message: str,
    details: dict | None = None,
    debug_info: str | None = None,
) -> JSONResponse:
    body: dict = {
        "success": False,
        "data": None,
        "message": message,
        "errors": details or {},
        "meta": {
            "request_id": request_id,
            "version": "1.0",
        },
    }
    if debug_info and get_settings().is_development:
        body["debug"] = debug_info
    return JSONResponse(status_code=status_code, content=body)


async def domain_exception_handler(request: Request, exc: AutoRecError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    if isinstance(exc, NotFoundError):
        status_code = status.HTTP_404_NOT_FOUND
    elif isinstance(exc, (InvalidFileTypeError, FileTooLargeError, ValidationError)):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, InvalidStateError):
        status_code = status.HTTP_409_CONFLICT
    elif isinstance(exc, (MLError, StorageError)):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        status_code = status.HTTP_400_BAD_REQUEST

    logger.warning(
        "domain_exception",
        request_id=request_id,
        path=str(request.url),
        error_type=type(exc).__name__,
        message=exc.message,
        details=exc.details,
    )

    return _error_response(
        request_id=request_id,
        status_code=status_code,
        message=exc.message,
        details=exc.details,
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))

    logger.error(
        "unhandled_exception",
        request_id=request_id,
        path=str(request.url),
        error_type=type(exc).__name__,
        exc_info=True,
    )

    return _error_response(
        request_id=request_id,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="An unexpected error occurred. Please try again or contact support.",
        debug_info=str(exc),
    )


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware that injects request_id into every request's state and
    logs all incoming requests with timing.
    """
    import time

    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    # Bind request_id to structlog context for all log calls in this request
    from structlog.contextvars import bind_contextvars, clear_contextvars
    clear_contextvars()
    bind_contextvars(request_id=request_id)

    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "http_request",
        method=request.method,
        path=str(request.url.path),
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )

    response.headers["X-Request-ID"] = request_id
    return response
