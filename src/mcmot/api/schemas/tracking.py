"""Response schemas for the tracking endpoints."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TrackingResponse", "TrackingStatus"]


class TrackingStatus(BaseModel):
    """One pipeline's state and rolling performance metrics."""

    # Permissive for the same reason as CameraResponse: the worker owns these
    # counters, and adding one should not mean editing the API layer.
    model_config = ConfigDict(extra="allow")

    camera_id: str
    state: str = Field(description="idle | running | stopped | failed")
    running: bool
    last_error: str | None = None
    fps: float = Field(
        description=(
            "Processed frames per second, measured to now — so it decays toward zero "
            "when a camera stalls rather than freezing at its last healthy value."
        )
    )
    latency_ms: float = Field(description="Mean end-to-end time per processed frame.")
    detect_ms: float
    reid_ms: float
    people: int = Field(description="Tracks in the most recent processed frame.")
    identified: int = Field(description="How many of those were positively named.")
    frames_processed: int
    frames_skipped_stale: int = Field(
        description="Frames already older than the staleness limit when reached."
    )
    embeddings_computed: int


class TrackingResponse(BaseModel):
    """Every running pipeline."""

    model_config = ConfigDict(extra="forbid")

    pipelines: list[TrackingStatus]
