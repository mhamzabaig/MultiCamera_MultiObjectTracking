"""Tracker output.

The contract between any tracker and everything downstream. ByteTrack,
BoT-SORT, and OC-SORT all reduce to a list of :class:`Track` objects per frame.

Note what is *absent*: no appearance feature, no identity, no name. A track is
pure short-term spatio-temporal continuity. Identity lives one layer up, in the
Identity Service, keyed by ``(camera_id, track_id)``. That separation is why a
ByteTrack ID switch after an occlusion costs a few frames of "Unknown" rather
than a wrong name: the new track simply re-queries the gallery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from mcmot.domain.geometry import BBox

__all__ = ["Track", "TrackState"]


class TrackState(str, Enum):
    """Lifecycle state of a track.

    Inherits from ``str`` so the value serialises directly to JSON and compares
    cleanly against configuration and API payloads.
    """

    TENTATIVE = "tentative"
    """Newly created, not yet confirmed by enough associations. Not displayed:
    tentative tracks are frequently false positives."""

    CONFIRMED = "confirmed"
    """Associated in the current frame. The only state that reaches the UI."""

    LOST = "lost"
    """Temporarily unmatched, retained for ``tracker.track_buffer`` frames so a
    brief occlusion does not force a new identity lookup."""

    REMOVED = "removed"
    """Exceeded the buffer and been retired. Its identity vote can be discarded."""


@dataclass(frozen=True, slots=True)
class Track:
    """One tracked object in one frame."""

    track_id: int
    """Tracker-assigned identifier, unique per camera, reused after a restart.
    Not a person identity — see the module docstring."""

    bbox: BBox
    score: float
    state: TrackState = TrackState.CONFIRMED
    age: int = 0
    """Frames elapsed since the track was created."""

    hits: int = 0
    """Number of frames in which this track was successfully associated."""

    time_since_update: int = 0
    """Frames since the last successful association; ``0`` when matched now."""

    class_id: int = 0

    @property
    def is_active(self) -> bool:
        """Whether the track was matched in the current frame."""
        return self.state is TrackState.CONFIRMED and self.time_since_update == 0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for logs and API responses."""
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox.to_xyxy()),
            "score": round(self.score, 4),
            "state": self.state.value,
            "age": self.age,
            "hits": self.hits,
            "time_since_update": self.time_since_update,
        }
