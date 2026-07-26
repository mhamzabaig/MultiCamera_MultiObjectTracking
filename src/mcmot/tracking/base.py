"""The tracker interface.

Takes the detections for one frame, returns :class:`~mcmot.domain.Track` objects
with stable ids across frames. No appearance model, no identity, no names — a
tracker provides short-term spatio-temporal continuity and nothing more. Identity
lives one layer up, keyed on ``(camera_id, track_id)``.

**A tracker is stateful and belongs to exactly one camera.** It holds Kalman filter
state per track, so instances are never shared between streams. The container
exposes a factory rather than a singleton for this reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mcmot.domain.detection import Detection
from mcmot.domain.track import Track

__all__ = ["BaseTracker"]


class BaseTracker(ABC):
    """Associates detections across frames for a single camera."""

    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance the tracker by one frame.

        Must be called once per processed frame, including frames with no
        detections — that is how tracks age out and how the frame counter stays
        aligned with reality.

        Args:
            detections: This frame's detections. May be empty.

        Returns:
            Currently confirmed tracks. Boxes are in the same pixel coordinates as
            the detections that produced them.
        """

    @abstractmethod
    def reset(self) -> None:
        """Discard all track state, e.g. after a camera reconnects.

        A reconnected stream may have skipped minutes of footage, so carrying
        Kalman predictions across the gap would produce confident nonsense.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary for the boot log and health endpoint."""
