"""YOLOv11 person detector, via Ultralytics.

Everything Ultralytics-specific is confined to this file. Three details are worth
knowing:

**Boxes come back in input-frame coordinates.** Ultralytics letterboxes the frame
to ``image_size`` internally and then maps boxes back itself, so ``result.boxes.xyxy``
is already in the original frame's pixel space. No manual rescaling — which is
where hand-rolled YOLO wrappers usually go wrong.

**Filtering happens inside the model call.** Passing ``conf``, ``iou``, and
``classes`` to ``predict`` lets NMS discard rejects on the GPU, rather than
returning 300 boxes for Python to filter.

**FP16 follows the device, not the config alone.** ``half=True`` on a CPU device is
slower than FP32, so the flag is the *conjunction* of the configured preference and
what :class:`~mcmot.core.device.DeviceManager` reports the hardware can actually do.

⚠️ Ultralytics is AGPL-3.0. See docs/THIRD_PARTY.md before shipping this
commercially.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mcmot.core.device import DeviceManager
from mcmot.core.errors import InvalidImageError, ModelLoadError
from mcmot.core.logging import get_logger
from mcmot.core.settings import DetectorSettings
from mcmot.detection.base import BaseDetector
from mcmot.domain.detection import Detection
from mcmot.domain.geometry import BBox
from mcmot.domain.types import ImageArray
from mcmot.ultralytics_guard import import_ultralytics

__all__ = ["YoloDetector"]

_log = get_logger(__name__)


class YoloDetector(BaseDetector):
    """Ultralytics YOLOv11 restricted to the configured classes."""

    def __init__(self, config: DetectorSettings, device_manager: DeviceManager) -> None:
        """Load the model.

        Args:
            config: Validated detector configuration.
            device_manager: Resolved compute device.

        Raises:
            ModelLoadError: If the weights are missing, corrupt, or incompatible.
        """
        self._config = config
        self._device = device_manager.info
        self._device_manager = device_manager
        self._half = config.half and self._device.fp16
        self._model = self._load()

    def _load(self) -> Any:
        weights = self._config.weights
        if not weights.is_file():
            raise ModelLoadError(
                f"Detector weights not found: {weights}. "
                "Run `python scripts/download_models.py` to fetch them.",
                details={"path": str(weights), "model": self._config.name},
            )

        # Guarded import: Ultralytics would otherwise pip-install missing
        # dependencies at runtime. See mcmot.ultralytics_guard.
        ultralytics = import_ultralytics("ultralytics")
        try:
            model = ultralytics.YOLO(str(weights), task="detect")
            model.to(self._device.torch_device)
        except Exception as exc:  # noqa: BLE001 - ultralytics raises broadly
            raise ModelLoadError(
                f"Could not load detector weights: {weights}",
                details={
                    "path": str(weights),
                    "device": self._device.torch_device,
                    "reason": str(exc),
                },
            ) from exc

        _log.info(
            "detector.loaded",
            model=self._config.name,
            weights=weights.name,
            device=self._device.torch_device,
            half=self._half,
            image_size=self._config.image_size,
            conf_threshold=self._config.conf_threshold,
        )
        return model

    # -- inference --------------------------------------------------------
    def detect(self, image: ImageArray) -> list[Detection]:
        """Detect people in one BGR frame. See :meth:`BaseDetector.detect`."""
        self._validate(image)

        results = self._model.predict(
            image,
            conf=self._config.conf_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.image_size,
            classes=list(self._config.classes),
            max_det=self._config.max_detections,
            half=self._half,
            device=self._device.torch_device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        # One host transfer for the whole frame rather than per box.
        coordinates = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        detections = [
            Detection(
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                # Clamp before constructing: Detection rejects out-of-range
                # scores, and float32 rounding can yield 1.0000001.
                score=float(min(max(score, 0.0), 1.0)),
                class_id=int(class_id),
            )
            for (x1, y1, x2, y2), score, class_id in zip(coordinates, scores, classes, strict=True)
        ]
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    def warmup(self) -> None:
        """Run inference on a synthetic frame. See :meth:`BaseDetector.warmup`."""
        dummy: ImageArray = np.zeros(
            (self._config.image_size, self._config.image_size, 3), dtype=np.uint8
        )
        self._device_manager.warmup(lambda: self.detect(dummy), label="detector")

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the loaded detector."""
        return {
            "name": self._config.name,
            "weights": self._config.weights.name,
            "device": self._device.torch_device,
            "half": self._half,
            "image_size": self._config.image_size,
            "conf_threshold": self._config.conf_threshold,
            "iou_threshold": self._config.iou_threshold,
            "classes": list(self._config.classes),
        }

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _validate(image: ImageArray) -> None:
        if image.ndim != 3 or image.shape[2] != 3:
            raise InvalidImageError(
                "Detector expects an HxWx3 BGR image.", details={"shape": list(image.shape)}
            )
        if image.dtype != np.uint8:
            raise InvalidImageError(
                "Detector expects a uint8 image.", details={"dtype": str(image.dtype)}
            )
        if image.size == 0:
            raise InvalidImageError("Detector received an empty image.")
