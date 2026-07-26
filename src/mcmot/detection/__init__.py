"""Person detection.

Deliberately not a plugin registry: a dict lookup over two-or-three known names is
easier to read and to debug than entry-point discovery. Adding RT-DETR later means
one new module and one entry in ``_DETECTORS``.
"""

from __future__ import annotations

from collections.abc import Callable

from mcmot.core.device import DeviceManager
from mcmot.core.errors import ConfigError
from mcmot.core.settings import DetectorSettings
from mcmot.detection.base import BaseDetector
from mcmot.detection.yolo import YoloDetector

__all__ = ["BaseDetector", "YoloDetector", "create_detector"]

#: Configured ``detector.name`` -> factory. Typed as a callable rather than a class
#: so an implementation is free to need different construction arguments.
_DETECTORS: dict[str, Callable[[DetectorSettings, DeviceManager], BaseDetector]] = {
    "yolo11": YoloDetector,
    # Ultralytics loads v8/v9/v10/v11 weights through the same class, so these are
    # aliases rather than separate implementations.
    "yolo": YoloDetector,
}


def create_detector(config: DetectorSettings, device_manager: DeviceManager) -> BaseDetector:
    """Build the detector named in configuration.

    Args:
        config: Validated detector configuration.
        device_manager: Resolved compute device.

    Returns:
        A ready detector. Call :meth:`BaseDetector.warmup` before the first frame.

    Raises:
        ConfigError: If ``detector.name`` is unknown.
        ModelLoadError: If the weights cannot be loaded.
    """
    try:
        factory = _DETECTORS[config.name.lower()]
    except KeyError:
        raise ConfigError(
            f"Unknown detector: {config.name!r}.",
            details={"name": config.name, "available": sorted(_DETECTORS)},
        ) from None
    return factory(config, device_manager)
