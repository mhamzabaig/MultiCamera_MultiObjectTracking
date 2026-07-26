"""Frame types carried through the tracking pipeline.

:class:`FrameMeta` is separated from the pixels on purpose. Metadata is small,
hashable, and safe to log, keep in a metrics ring buffer, or ship over a queue;
the pixel array is large and must never be copied casually. Keeping them
separate makes it obvious in a signature which one a function actually needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mcmot.domain.types import ImageArray

__all__ = ["Frame", "FrameMeta"]


@dataclass(frozen=True, slots=True)
class FrameMeta:
    """Provenance and timing for a single captured frame."""

    camera_id: str
    index: int
    """Monotonic per-camera counter. Gaps are expected and meaningful: they are
    exactly the frames that latest-frame-wins capture dropped."""

    timestamp: float
    """Capture time as a UNIX epoch (seconds, UTC), from ``time.time()``."""

    width: int
    height: int
    monotonic: float = field(default_factory=time.monotonic)
    """Capture time on the monotonic clock. All latency arithmetic uses this;
    ``timestamp`` is for display only, since a clock adjustment mid-stream would
    otherwise produce negative or absurd latencies."""

    def age_s(self, now: float | None = None) -> float:
        """Seconds since capture, measured on the monotonic clock."""
        return (time.monotonic() if now is None else now) - self.monotonic

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for logs and API responses."""
        return {
            "camera_id": self.camera_id,
            "index": self.index,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """A decoded frame: BGR pixels plus their metadata.

    ``eq=False`` because the payload is a NumPy array - a generated ``__eq__``
    would return an array and raise on truth-testing. Frames are identified by
    ``meta``, never compared by content.

    The array is **not** copied and must be treated as read-only. Capture threads
    hand ownership downstream and immediately reuse their buffers; a stage that
    needs to draw must call :meth:`copy_image` first.
    """

    image: ImageArray
    """BGR ``uint8`` array of shape ``(height, width, 3)`` - OpenCV's layout."""

    meta: FrameMeta

    def __post_init__(self) -> None:
        """Validate shape and dtype once, at the pipeline boundary."""
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"Expected an HxWx3 BGR image, got shape {self.image.shape!r}")
        if self.image.dtype != np.uint8:
            raise ValueError(f"Expected a uint8 image, got dtype {self.image.dtype}")

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(height, width)``."""
        return int(self.image.shape[0]), int(self.image.shape[1])

    def copy_image(self) -> ImageArray:
        """Return a writable copy, for stages that annotate in place."""
        return self.image.copy()

    @classmethod
    def create(
        cls,
        image: ImageArray,
        *,
        camera_id: str,
        index: int,
        timestamp: float | None = None,
    ) -> Frame:
        """Build a frame, deriving dimensions from the array.

        Args:
            image: BGR ``uint8`` array.
            camera_id: Identifier of the originating camera.
            index: Monotonic per-camera frame counter.
            timestamp: Capture epoch; defaults to now.
        """
        height, width = int(image.shape[0]), int(image.shape[1])
        return cls(
            image=image,
            meta=FrameMeta(
                camera_id=camera_id,
                index=index,
                timestamp=time.time() if timestamp is None else timestamp,
                width=width,
                height=height,
            ),
        )
