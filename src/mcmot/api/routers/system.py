"""System endpoints: health, version, configuration, logs.

Every router follows this file's shape: a thin handler that validates input,
delegates to a service obtained by dependency injection, and returns a declared
response model. No business logic lives in the API layer.

Settings and logs live here rather than in routers of their own because both are
process-level reads that need nothing but the container — a router per endpoint
would be structure for its own sake.
"""

from __future__ import annotations

import json
from collections import deque
from typing import Annotated, Literal

from fastapi import APIRouter, Query, status

from mcmot.api.deps import ContainerDep, SettingsDep
from mcmot.api.schemas.system import (
    ErrorResponse,
    HealthResponse,
    LogEntry,
    LogsResponse,
    SettingsResponse,
    VersionResponse,
)
from mcmot.core.settings import resolve_config_dir

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


@router.get(
    "/settings",
    response_model=SettingsResponse,
    summary="Active configuration, redacted and read-only",
)
def read_settings(settings: SettingsDep) -> SettingsResponse:
    """Return the effective configuration.

    Read-only: see :class:`SettingsResponse` for why. The payload is the same
    redacted summary the boot log emits, so what the page shows and what the log
    recorded cannot drift apart.
    """
    config_dir = resolve_config_dir()
    return SettingsResponse(
        config=settings.summary(),
        editable=False,
        # Merge order, lowest precedence first — the order in which a reader should
        # expect a value to be overridden.
        config_files=[
            str(config_dir / name)
            for name in ("default.yaml", "local.yaml")
            if (config_dir / name).exists()
        ],
    )


@router.get("/logs", response_model=LogsResponse, summary="Tail the structured log")
def read_logs(
    container: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=2000, description="How many records to return.")] = 200,
    level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"] | None,
        Query(description="Return only records at or above this level."),
    ] = None,
) -> LogsResponse:
    """Return the tail of the JSON log file.

    Reads the file rather than keeping a ring buffer in memory: the rotating JSON
    sink is already the durable record, and a second in-process copy would disagree
    with it after a restart and cost memory to maintain.

    Malformed lines are skipped. A log file being written to while it is read will
    occasionally yield a half-flushed final line, and refusing to show any logs
    because of one truncated record would be exactly backwards.
    """
    settings = container.settings
    if not settings.logging.file_enabled:
        return LogsResponse(entries=[], file=None, truncated=False)

    path = settings.log_file_path
    if not path.exists():
        # Nothing logged to file yet — a fresh process, not an error.
        return LogsResponse(entries=[], file=str(path), truncated=False)

    wanted = _LEVELS.index(level.upper()) if level else 0
    # A bounded deque means memory is capped by `limit`, not by the file size, so a
    # 10 MB log costs the same as an empty one.
    kept: deque[LogEntry] = deque(maxlen=limit)
    seen = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = _parse_line(line)
            if record is None:
                continue
            if wanted and _rank(record.get("level")) < wanted:
                continue
            seen += 1
            kept.append(LogEntry.model_validate(record))

    return LogsResponse(entries=list(kept), file=str(path), truncated=seen > len(kept))


#: Ordered so a numeric comparison can express "at or above this level".
_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _rank(level: object) -> int:
    """Return a level's severity rank; unknown levels sort lowest."""
    if not isinstance(level, str):
        return 0
    try:
        return _LEVELS.index(level.upper())
    except ValueError:
        return 0


def _parse_line(line: str) -> dict[str, object] | None:
    """Parse one JSON log line, or ``None`` if it is not a usable record."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        record = json.loads(stripped)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None
