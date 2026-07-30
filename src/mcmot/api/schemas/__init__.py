"""Pydantic request and response schemas for the HTTP API."""

from __future__ import annotations

from mcmot.api.schemas.cameras import CameraResponse, CamerasResponse, OpenCameraRequest
from mcmot.api.schemas.persons import (
    GalleryStats,
    PersonResponse,
    PersonsResponse,
    RebuildResponse,
    RenameRequest,
)
from mcmot.api.schemas.system import (
    DeviceStatus,
    ErrorResponse,
    HealthResponse,
    LogEntry,
    LogsResponse,
    SettingsResponse,
    VersionResponse,
)
from mcmot.api.schemas.tracking import TrackingResponse, TrackingStatus

__all__ = [
    "CameraResponse",
    "CamerasResponse",
    "DeviceStatus",
    "ErrorResponse",
    "GalleryStats",
    "HealthResponse",
    "LogEntry",
    "LogsResponse",
    "OpenCameraRequest",
    "PersonResponse",
    "PersonsResponse",
    "RebuildResponse",
    "RenameRequest",
    "SettingsResponse",
    "TrackingResponse",
    "TrackingStatus",
    "VersionResponse",
]
