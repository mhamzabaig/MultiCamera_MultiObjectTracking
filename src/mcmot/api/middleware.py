"""HTTP middleware: request correlation and access logging.

Every request gets an id that appears in the response header, in every log line
emitted while handling it, and in any error payload returned. When a camera
misbehaves and the user reports "it failed at about 3pm", that id is what turns a
vague complaint into one grep.

Access logging lives here rather than in uvicorn's own logger so that requests
arrive in the same structured stream as everything else, with duration attached.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from mcmot.core.logging import bind_context, get_logger

__all__ = ["REQUEST_ID_HEADER", "RequestContextMiddleware"]

REQUEST_ID_HEADER = "X-Request-ID"

_log = get_logger(__name__)

#: Endpoints excluded from access logging. The MJPEG stream is a single
#: long-lived request per viewer and health is polled continuously; logging
#: either would drown the signal in this stream.
_QUIET_PATHS = frozenset({"/api/system/health", "/favicon.ico"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to the log context, and time the request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Wrap one request with correlation context and an access log line."""
        # Honour an inbound id so a correlation started by a proxy or the
        # frontend survives the hop.
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        started = time.perf_counter()
        with bind_context(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception:
                # The exception handlers produce the client response; this exists
                # so the failure is timed and correlated even when it escapes.
                _log.exception(
                    "http.request_failed",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
                raise

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers[REQUEST_ID_HEADER] = request_id

            if request.url.path not in _QUIET_PATHS:
                log = _log.warning if response.status_code >= 400 else _log.info
                log(
                    "http.request",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    duration_ms=duration_ms,
                )
            return response
