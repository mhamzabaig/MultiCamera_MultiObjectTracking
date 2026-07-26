"""Application factory and process entry point.

``create_app`` is a factory, not a module-level ``app = FastAPI()``. That means a
test can build an app around a container wired with temporary directories and
fake collaborators, and several apps can coexist in one interpreter. A
module-level ``app`` is still exported for ``uvicorn mcmot.main:app``, but it is
constructed on import only when actually referenced by the server.
"""

from __future__ import annotations

import argparse
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mcmot.api.errors import register_exception_handlers
from mcmot.api.middleware import RequestContextMiddleware
from mcmot.api.routers import all_routers
from mcmot.core.container import Container
from mcmot.core.logging import get_logger
from mcmot.core.settings import Settings, load_settings

__all__ = ["cli", "create_app"]

_log = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    container: Container | None = None,
) -> FastAPI:
    """Build the ASGI application.

    Args:
        settings: Configuration to use; loaded from the standard layers when
            omitted. Ignored if ``container`` is supplied.
        container: Pre-wired container, for tests that inject fakes.

    Returns:
        A configured FastAPI application.
    """
    app_container = container if container is not None else Container(settings)
    config = app_container.settings

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Own the container's lifetime, tied to the server's.

        A failure in ``startup`` propagates and aborts the boot on purpose: a
        server that accepts requests with an unwritable log directory or an
        unusable GPU would only fail later, less clearly.
        """
        app_container.startup()
        try:
            yield
        finally:
            app_container.shutdown()

    app = FastAPI(
        title="MC-MOT",
        summary="Multi-Camera Multi-Object Tracking and Person Re-Identification",
        version=config.version,
        lifespan=lifespan,
        docs_url="/docs" if config.api.docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if config.api.docs_enabled else None,
    )
    app.state.container = app_container

    # Order matters: middleware added last runs outermost, so request-id binding
    # must be registered after CORS to wrap it and cover preflight failures too.
    if config.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.api.cors_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    for router in all_routers():
        app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        """Service descriptor. Phase 10 replaces this with the dashboard page."""
        return JSONResponse(
            {
                "name": "mcmot",
                "version": config.version,
                "docs": "/docs" if config.api.docs_enabled else None,
                "health": "/api/system/health",
            }
        )

    return app


def _build_default_app() -> FastAPI:
    return create_app()


def cli(argv: list[str] | None = None) -> int:
    """Run the development server.

    Reads host and port from configuration so that ``api.host`` / ``api.port``
    stay the single source of truth, with command-line flags available for
    one-off overrides.

    Args:
        argv: Argument vector; ``sys.argv[1:]`` when omitted.

    Returns:
        Process exit code.
    """
    import uvicorn  # noqa: PLC0415 - only needed when actually serving

    parser = argparse.ArgumentParser(prog="mcmot", description="Run the MC-MOT server.")
    parser.add_argument("--host", default=None, help="Bind address (overrides config).")
    parser.add_argument("--port", type=int, default=None, help="Port (overrides config).")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on source changes (development only).",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    host = args.host or settings.api.host
    port = args.port or settings.api.port

    _log.info("server.launching", host=host, port=port, reload=args.reload)

    kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        # Our own middleware emits structured access logs; uvicorn's duplicate
        # them in a different format.
        "access_log": False,
        "log_level": settings.logging.level.lower(),
    }
    if args.reload:
        # The reloader re-imports the app in a child process, so it needs an
        # import string rather than an object.
        uvicorn.run("mcmot.main:app", reload=True, **kwargs)
    else:
        uvicorn.run(create_app(settings), **kwargs)
    return 0


def __getattr__(name: str) -> Any:
    """Construct the module-level ``app`` lazily.

    Keeps ``import mcmot.main`` free of side effects — importing a module should
    not read configuration or touch the filesystem — while still supporting
    ``uvicorn mcmot.main:app``.
    """
    if name == "app":
        app = _build_default_app()
        globals()["app"] = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
