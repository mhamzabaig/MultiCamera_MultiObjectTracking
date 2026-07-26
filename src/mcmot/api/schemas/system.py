"""Response schemas for the system endpoints.

Declared as explicit Pydantic models rather than returned as bare dictionaries so
that the OpenAPI schema at ``/docs`` is accurate and the frontend has a contract
it can rely on. ``extra="forbid"`` on responses catches a field we forgot to
document as a test failure rather than as an undocumented API.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["DeviceStatus", "ErrorResponse", "HealthResponse", "VersionResponse"]


class DeviceStatus(BaseModel):
    """Resolved compute device and its capabilities."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["cuda", "cpu"] = Field(description="Resolved device type.")
    index: int | None = Field(default=None, description="CUDA ordinal, null on CPU.")
    name: str = Field(description="Human-readable device name.")
    fp16: bool = Field(description="Whether half precision is in use.")
    total_vram_mb: int | None = None
    free_vram_mb: int | None = None
    capability: str | None = Field(default=None, description='CUDA compute capability, e.g. "8.9".')
    torch_version: str | None = None
    cuda_version: str | None = None
    is_fallback: bool = Field(description="True when the active device is not the one requested.")
    reason: str | None = Field(
        default=None, description="Why a fallback occurred; null on clean resolution."
    )
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness, readiness, and the effective configuration."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded", "unhealthy"] = Field(
        description=(
            "'degraded' means serving but not at full capability — typically a CPU "
            "fallback after CUDA was found unusable."
        )
    )
    version: str
    env: Literal["dev", "test", "prod"]
    uptime_s: float
    started_epoch: float | None
    device: DeviceStatus
    config: dict[str, Any] = Field(description="Redacted summary of the active configuration.")


class VersionResponse(BaseModel):
    """Build identity, for cheap polling that should not touch the device."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    env: Literal["dev", "test", "prod"]


class ErrorResponse(BaseModel):
    """RFC 7807-style error payload returned by every failing endpoint."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description='Stable machine-readable slug, e.g. "camera.timeout".')
    message: str
    status: int
    details: dict[str, Any] | None = None
    request_id: str | None = Field(
        default=None, description="Correlation id; matches the X-Request-ID header."
    )
