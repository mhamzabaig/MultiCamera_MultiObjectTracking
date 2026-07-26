"""Crop quality gate.

Rejects person crops that are not worth embedding: too small, wrongly shaped, half
outside the frame, or motion-blurred.

This is not a performance optimisation — it is an accuracy mechanism. A bad crop
still produces a confident-looking 512-dimensional vector, and that vector goes into
the same temporal vote as the good ones. Half a body or a smear of motion blur
resembles *every* person about equally, so admitting it drags a track's accumulated
evidence toward whichever gallery entry happens to be closest to the average. The
cheapest way to keep the vote honest is to never let those crops in.

The same gate serves both directions, which is the point: registration selects its
best frames with it, and tracking filters with it, so enrolment and inference agree
on what a usable crop is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from mcmot.core.settings import QualitySettings
from mcmot.domain.geometry import BBox
from mcmot.domain.types import ImageArray

__all__ = ["CropQuality", "QualityGate"]


@dataclass(frozen=True, slots=True)
class CropQuality:
    """The verdict on one crop, with the measurements behind it."""

    accepted: bool
    score: float
    """Overall quality in ``[0, 1]``. Used to rank candidate registration frames and
    to decide which embedding to evict when a person exceeds their gallery limit."""

    reason: str | None = None
    """Why it was rejected, or ``None`` when accepted. Retained because "recognition
    stopped working" is usually "every crop is being rejected", and this names which
    check is responsible."""

    width: int = 0
    height: int = 0
    aspect_ratio: float = 0.0
    blur: float = 0.0
    """Variance of the Laplacian. Higher is sharper."""

    truncation: float = 0.0
    """Fraction of the box outside the frame."""

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for logs and debug views."""
        return {
            "accepted": self.accepted,
            "score": round(self.score, 4),
            "reason": self.reason,
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 4),
            "blur": round(self.blur, 2),
            "truncation": round(self.truncation, 4),
        }


class QualityGate:
    """Assesses whether a person crop is worth embedding."""

    def __init__(self, config: QualitySettings) -> None:
        """Args: config: Validated crop-quality configuration."""
        self._config = config

    def assess(
        self,
        crop: ImageArray,
        *,
        bbox: BBox | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> CropQuality:
        """Judge one crop.

        Checks run cheapest-first and stop at the first failure, so a 20×30 crop
        never pays for a Laplacian.

        Args:
            crop: The cropped region, BGR ``uint8``.
            bbox: The detection box in frame coordinates. Required together with
                ``frame_shape`` to measure truncation, since a crop that has already
                been clipped to the frame carries no evidence that it was cut off.
            frame_shape: ``(height, width)`` of the source frame.

        Returns:
            The verdict and its measurements.
        """
        if crop.size == 0 or crop.ndim != 3:
            return CropQuality(accepted=False, score=0.0, reason="empty_crop")

        height, width = int(crop.shape[0]), int(crop.shape[1])
        aspect = width / height if height > 0 else float("inf")

        if width < self._config.min_width or height < self._config.min_height:
            return CropQuality(
                accepted=False,
                score=0.0,
                reason="too_small",
                width=width,
                height=height,
                aspect_ratio=aspect,
            )

        if not self._config.min_aspect_ratio <= aspect <= self._config.max_aspect_ratio:
            # Usually a merged detection covering two people, or a reflection.
            return CropQuality(
                accepted=False,
                score=0.0,
                reason="bad_aspect_ratio",
                width=width,
                height=height,
                aspect_ratio=aspect,
            )

        truncation = 0.0
        if bbox is not None and frame_shape is not None:
            truncation = bbox.truncation(frame_shape[1], frame_shape[0])
            if truncation > self._config.max_truncation:
                return CropQuality(
                    accepted=False,
                    score=0.0,
                    reason="truncated",
                    width=width,
                    height=height,
                    aspect_ratio=aspect,
                    truncation=truncation,
                )

        blur = self._sharpness(crop)
        if blur < self._config.blur_threshold:
            return CropQuality(
                accepted=False,
                score=0.0,
                reason="too_blurry",
                width=width,
                height=height,
                aspect_ratio=aspect,
                blur=blur,
                truncation=truncation,
            )

        return CropQuality(
            accepted=True,
            score=self._score(width, height, blur, truncation),
            width=width,
            height=height,
            aspect_ratio=aspect,
            blur=blur,
            truncation=truncation,
        )

    # -- measurements -----------------------------------------------------
    @staticmethod
    def _sharpness(crop: ImageArray) -> float:
        """Return the variance of the Laplacian: low means blurred.

        Computed on a downscaled grayscale copy. Full resolution would make this the
        most expensive check in the gate for no gain, since blur is a low-frequency
        property that survives downscaling.
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if gray.shape[0] > 128:
            scale = 128 / gray.shape[0]
            gray = cv2.resize(
                gray, (max(1, int(gray.shape[1] * scale)), 128), interpolation=cv2.INTER_AREA
            )
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _score(self, width: int, height: int, blur: float, truncation: float) -> float:
        """Combine the measurements into a single ``[0, 1]`` quality score.

        Only ever used to *rank* crops against each other — never as a threshold.
        The thresholds are the individual checks above, which are interpretable and
        independently tunable.

        Both components use ``x / (x + k)`` rather than a clipped ratio. A clipped
        ratio was measured returning exactly 1.000 for every crop cut from a real
        720p frame — real detections are far above the rejection floors (blur ~4000
        against a threshold of 45), so a hard ceiling made the score constant and
        therefore useless for choosing the best registration frames. This form
        saturates smoothly and never quite reaches 1.0, so ordering is preserved
        across the entire range.
        """
        area = float(width * height)
        reference_area = float(self._config.min_width * self._config.min_height * 4)
        size_score = area / (area + reference_area)

        reference_blur = float(self._config.blur_threshold * 4)
        blur_score = blur / (blur + reference_blur) if blur > 0 else 0.0

        visibility = 1.0 - float(np.clip(truncation, 0.0, 1.0))
        return round(0.4 * size_score + 0.4 * blur_score + 0.2 * visibility, 6)
