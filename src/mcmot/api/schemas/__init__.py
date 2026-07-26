"""Pydantic request and response schemas for the HTTP API."""

from __future__ import annotations

from mcmot.api.schemas.system import (
    DeviceStatus,
    ErrorResponse,
    HealthResponse,
    VersionResponse,
)

__all__ = ["DeviceStatus", "ErrorResponse", "HealthResponse", "VersionResponse"]
