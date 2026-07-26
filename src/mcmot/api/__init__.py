"""HTTP API layer: routers, schemas, dependencies, middleware, error mapping."""

from __future__ import annotations

from mcmot.api.deps import ContainerDep, SettingsDep, get_container, get_settings
from mcmot.api.errors import register_exception_handlers
from mcmot.api.middleware import REQUEST_ID_HEADER, RequestContextMiddleware
from mcmot.api.routers import all_routers

__all__ = [
    "REQUEST_ID_HEADER",
    "ContainerDep",
    "RequestContextMiddleware",
    "SettingsDep",
    "all_routers",
    "get_container",
    "get_settings",
    "register_exception_handlers",
]
