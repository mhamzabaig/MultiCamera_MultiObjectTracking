"""The detector interface.

One method that matters: give it a BGR frame, get back a list of
:class:`~mcmot.domain.Detection`. No framework types cross this boundary — no
Ultralytics ``Results``, no tensors — which is what makes YOLOv11 replaceable by
RT-DETR or YOLO-NAS without touching the tracker, the pipeline, or the UI.

An abstract base class rather than a ``Protocol`` because there is real shared
behaviour to inherit: the warm-up contract and the ``describe()`` reporting used by
the health endpoint and the boot log.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mcmot.domain.detection import Detection
from mcmot.domain.types import ImageArray

__all__ = ["BaseDetector"]


class BaseDetector(ABC):
    """Finds people in a single frame."""

    @abstractmethod
    def detect(self, image: ImageArray) -> list[Detection]:
        """Detect people in one BGR frame.

        Implementations must:

        * return boxes in **absolute pixel coordinates of the input frame**, so a
          caller never has to know what resolution the model ran at;
        * apply the configured confidence and IoU thresholds themselves;
        * return only the configured classes;
        * return an empty list when nothing is found — never ``None``.

        Args:
            image: BGR ``uint8`` array of shape ``(height, width, 3)``.

        Returns:
            Detections, ordered by descending confidence.

        Raises:
            InvalidImageError: If the frame is not a usable BGR image.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Run inference on a dummy frame to absorb one-off initialisation.

        The first CUDA forward pass pays for context creation, kernel autotuning,
        and cuDNN algorithm selection — commonly 1–2 seconds. Paying it at startup
        keeps it out of the first tracked frame, where it would appear as a stall
        and a dropped stream.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary for the boot log and health endpoint."""
