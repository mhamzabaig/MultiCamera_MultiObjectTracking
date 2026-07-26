"""Translation of exceptions into HTTP responses.

Every failure leaves the API in one shape — ``{code, message, status, details,
request_id}`` — regardless of whether it originated in our code, in Pydantic
validation, or in an unhandled bug. A frontend switches on ``code``; a human
quotes ``request_id``.

Two rules:

* **Domain errors carry their own status.** :class:`~mcmot.core.errors.MCMOTError`
  subclasses declare ``code`` and ``http_status`` next to the failure they
  describe, so there is no status-mapping table here to drift out of date.
* **Unexpected exceptions never leak internals.** The traceback goes to the log
  with the request id attached; the client gets a generic 500 and that id. A
  stack trace in a browser response is an information-disclosure bug.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from mcmot.core.errors import MCMOTError
from mcmot.core.logging import get_logger

__all__ = ["register_exception_handlers"]

_log = get_logger(__name__)

#: RFC 7807 media type, so clients can distinguish an error body from a payload.
_PROBLEM_JSON = "application/problem+json"


def _request_id(request: Request) -> str | None:
    value = getattr(request.state, "request_id", None)
    return str(value) if value else None


def _problem(
    request: Request,
    *,
    status: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    body: dict[str, Any] = {"code": code, "message": message, "status": int(status)}
    if details:
        body["details"] = details
    request_id = _request_id(request)
    if request_id:
        body["request_id"] = request_id
    return JSONResponse(status_code=int(status), content=body, media_type=_PROBLEM_JSON)


def register_exception_handlers(app: FastAPI) -> None:
    """Install the handlers on ``app``. Called once by the app factory."""

    @app.exception_handler(MCMOTError)
    async def _handle_domain_error(request: Request, exc: MCMOTError) -> JSONResponse:
        # Expected, described failures: log at warning with structure, no traceback.
        _log.warning(
            "api.domain_error",
            code=exc.code,
            status=int(exc.http_status),
            path=request.url.path,
            **exc.details,
        )
        return _problem(
            request,
            status=exc.http_status,
            code=exc.code,
            message=exc.message,
            details=exc.details or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        _log.warning(
            "api.validation_error",
            path=request.url.path,
            error_count=len(exc.errors()),
        )
        return _problem(
            request,
            status=HTTPStatus.UNPROCESSABLE_ENTITY,
            code="request.invalid",
            message="Request validation failed.",
            # Pydantic's error objects can hold non-serialisable values under
            # "input"; keep only the fields a client can act on.
            details={
                "errors": [
                    {
                        "location": list(error.get("loc", ())),
                        "message": error.get("msg", ""),
                        "type": error.get("type", ""),
                    }
                    for error in exc.errors()
                ]
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        try:
            phrase = HTTPStatus(exc.status_code).phrase.lower().replace(" ", "_")
        except ValueError:  # pragma: no cover - non-standard status code
            phrase = "error"
        return _problem(
            request,
            status=exc.status_code,
            code=f"http.{phrase}",
            message=str(exc.detail),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        # A genuine bug: full traceback to the log, nothing internal to the client.
        _log.exception(
            "api.unhandled_exception",
            path=request.url.path,
            exception_type=type(exc).__name__,
        )
        return _problem(
            request,
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            code="internal.error",
            message="An unexpected internal error occurred.",
        )
