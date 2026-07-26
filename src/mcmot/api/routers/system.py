"""System endpoints: health, version, configuration.

The routers added in phase 9 (persons, cameras, tracking, settings, logs) follow
this file's shape: a thin handler that validates input, delegates to a service
obtained by dependency injection, and returns a declared response model. No
business logic lives in the API layer.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from mcmot.api.deps import ContainerDep, SettingsDep
from mcmot.api.schemas.system import ErrorResponse, HealthResponse, VersionResponse

__all__ = ["router"]

router = APIRouter(
    prefix="/api/system",
    tags=["system"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Unexpected internal error.",
        }
    },
)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness, readiness, and effective configuration",
)
async def health(container: ContainerDep) -> HealthResponse:
    """Report process health, the resolved compute device, and active thresholds.

    Always answers ``200`` while the process is serving. Degradation is expressed
    in the ``status`` field rather than the HTTP code, because a CPU fallback is a
    working system that a monitor should distinguish from an outage — collapsing
    both into ``503`` would lose that distinction.
    """
    return HealthResponse.model_validate(container.health())


@router.get(
    "/version",
    response_model=VersionResponse,
    summary="Build identity",
)
async def version(settings: SettingsDep) -> VersionResponse:
    """Return name, version, and environment.

    Deliberately cheap: it touches no device and no storage, so it is the right
    endpoint for a frequently-polled uptime check.
    """
    return VersionResponse(name="mcmot", version=settings.version, env=settings.env)
