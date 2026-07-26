"""Detector output.

The contract between any detector and everything downstream. A detector returns
:class:`Detection` objects and nothing else — no framework-specific result
wrappers, no tensors — which is what makes YOLOv11 swappable for RT-DETR or
YOLO-NAS without touching the tracker, the pipeline, or the tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mcmot.domain.geometry import BBox

__all__ = ["Detection"]

#: COCO class index for "person" — the only class this system tracks.
PERSON_CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected object in one frame."""

    bbox: BBox
    score: float
    """Detector confidence in ``[0, 1]``."""

    class_id: int = PERSON_CLASS_ID

    def __post_init__(self) -> None:
        """Guard the confidence range, so downstream thresholds are meaningful."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Detection score must lie in [0, 1], got {self.score}")

    @property
    def is_person(self) -> bool:
        """Whether this detection is of the person class."""
        return self.class_id == PERSON_CLASS_ID

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for logs and API responses."""
        return {
            "bbox": list(self.bbox.to_xyxy()),
            "score": round(self.score, 4),
            "class_id": self.class_id,
        }
