"""Request and response schemas for the cameras endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CameraResponse", "CamerasResponse", "OpenCameraRequest"]


class OpenCameraRequest(BaseModel):
    """A camera to open."""

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
        description=(
            "Stable identifier, unique among open cameras. Restricted to path-safe "
            "characters because it appears in URLs and in log context."
        ),
    )
    source: str = Field(
        min_length=1,
        description=(
            'Webcam index ("0"), rtsp:// URL, http:// MJPEG URL (what phone '
            "IP-camera apps serve), or a path to a video file."
        ),
    )


class CameraResponse(BaseModel):
    """A camera's specification and live capture statistics."""

    # Deliberately permissive: the capture thread owns this payload and adding a
    # counter there should not require editing a schema in the API layer.
    model_config = ConfigDict(extra="allow")

    camera_id: str
    source: str = Field(description="Source string with any credentials redacted.")
    kind: str = Field(description="webcam | rtsp | http | file")
    state: str = Field(description="idle | connecting | streaming | reconnecting | ended | failed")
    running: bool
    fps: float = Field(description="Capture rate over the recent window.")
    frames_delivered: int = Field(description="Frames captured and published.")
    frames_dropped: int = Field(description="Frames replaced before anything read them.")
    reconnects: int
    last_error: str | None = None
    frame_index: int
    latency_ms: float | None = Field(
        default=None, description="Age of the newest frame; null if none has arrived."
    )


class CamerasResponse(BaseModel):
    """Every open camera."""

    model_config = ConfigDict(extra="forbid")

    cameras: list[CameraResponse]
