"""Geometric primitives shared by detection, tracking, and re-identification.

One bounding-box representation for the whole system. Detectors, trackers, crop
extraction, the quality gate, and the annotator all speak :class:`BBox`, so the
classic source of silent bugs — ``xyxy`` from one stage meeting ``xywh`` in the
next — cannot occur. Conversions are explicit, named methods.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BBox"]


@dataclass(frozen=True, slots=True)
class BBox:
    """An axis-aligned bounding box in absolute pixel coordinates.

    Canonical form is ``(x1, y1, x2, y2)`` with the origin at the top-left of the
    frame, ``x2 >= x1`` and ``y2 >= y1``. Coordinates are floats: trackers
    predict sub-pixel positions, and rounding only at the drawing and cropping
    boundaries avoids accumulating error through the Kalman filter.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        """Reject inverted boxes at construction rather than downstream."""
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(
                f"Invalid bounding box: ({self.x1}, {self.y1}, {self.x2}, {self.y2}); "
                "expected x2 >= x1 and y2 >= y1"
            )

    # -- measurements -----------------------------------------------------
    @property
    def width(self) -> float:
        """Box width in pixels."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Width divided by height; ``inf`` for a zero-height box.

        Used by the ReID quality gate: a person crop far outside the expected
        range is usually a merged detection or a reflection, not a person.
        """
        return self.width / self.height if self.height > 0 else float("inf")

    @property
    def center(self) -> tuple[float, float]:
        """Box centre as ``(cx, cy)``."""
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    # -- conversions ------------------------------------------------------
    def to_xyxy(self) -> tuple[float, float, float, float]:
        """Return ``(x1, y1, x2, y2)``."""
        return self.x1, self.y1, self.x2, self.y2

    def to_xywh(self) -> tuple[float, float, float, float]:
        """Return ``(x, y, width, height)`` with ``(x, y)`` at the top-left."""
        return self.x1, self.y1, self.width, self.height

    def to_cxcywh(self) -> tuple[float, float, float, float]:
        """Return ``(cx, cy, width, height)`` — the form Kalman filters expect."""
        cx, cy = self.center
        return cx, cy, self.width, self.height

    def to_int(self) -> tuple[int, int, int, int]:
        """Return integer ``(x1, y1, x2, y2)`` for slicing and drawing.

        Rounds outward so a crop never loses a boundary pixel present in the
        float box.
        """
        import math  # noqa: PLC0415 - trivial, keeps module import surface small

        return (
            int(math.floor(self.x1)),
            int(math.floor(self.y1)),
            int(math.ceil(self.x2)),
            int(math.ceil(self.y2)),
        )

    @classmethod
    def from_xywh(cls, x: float, y: float, width: float, height: float) -> BBox:
        """Build a box from top-left corner plus size."""
        return cls(x, y, x + width, y + height)

    @classmethod
    def from_cxcywh(cls, cx: float, cy: float, width: float, height: float) -> BBox:
        """Build a box from centre plus size."""
        half_w, half_h = width / 2.0, height / 2.0
        return cls(cx - half_w, cy - half_h, cx + half_w, cy + half_h)

    # -- operations -------------------------------------------------------
    def clip(self, frame_width: int, frame_height: int) -> BBox:
        """Clamp the box to the frame, so cropping can never index out of bounds."""
        return BBox(
            max(0.0, min(self.x1, float(frame_width))),
            max(0.0, min(self.y1, float(frame_height))),
            max(0.0, min(self.x2, float(frame_width))),
            max(0.0, min(self.y2, float(frame_height))),
        )

    def truncation(self, frame_width: int, frame_height: int) -> float:
        """Return the fraction of the box lying outside the frame, in ``[0, 1]``.

        A person half out of view yields a crop containing half a body, whose
        embedding is unreliable. The quality gate rejects crops above a
        configured threshold rather than letting them corrupt the identity vote.
        """
        if self.area <= 0:
            return 1.0
        visible = self.clip(frame_width, frame_height).area
        return max(0.0, 1.0 - visible / self.area)

    def expand(self, ratio: float) -> BBox:
        """Grow the box about its centre by ``ratio`` (``0.1`` == 10% larger).

        A small margin around a detection improves ReID accuracy: tight boxes
        clip shoulders and shoes, and the resulting crops differ systematically
        from the enrolment crops.
        """
        if ratio <= -1.0:
            raise ValueError("ratio must be greater than -1.0")
        cx, cy = self.center
        return BBox.from_cxcywh(cx, cy, self.width * (1.0 + ratio), self.height * (1.0 + ratio))

    def iou(self, other: BBox) -> float:
        """Intersection over union with ``other``, in ``[0, 1]``."""
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter_w, inter_h = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        intersection = inter_w * inter_h
        if intersection <= 0.0:
            return 0.0
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0
