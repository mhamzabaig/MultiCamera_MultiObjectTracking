"""Person re-identification.

Converts a person crop into an appearance fingerprint. Same shape as
:mod:`mcmot.detection` and :mod:`mcmot.tracking`: a dict lookup, not a plugin
registry. Adding a FastReID or TransReID backend later means one new module and one
entry in ``_EMBEDDERS``.
"""

from __future__ import annotations

from collections.abc import Callable

from mcmot.core.device import DeviceManager
from mcmot.core.errors import ConfigError
from mcmot.core.settings import ReIDSettings
from mcmot.reid.base import BaseEmbedder
from mcmot.reid.embedder import OsNetEmbedder
from mcmot.reid.quality import CropQuality, QualityGate

__all__ = [
    "BaseEmbedder",
    "CropQuality",
    "OsNetEmbedder",
    "QualityGate",
    "create_embedder",
]

#: Configured ``reid.name`` -> factory.
_EMBEDDERS: dict[str, Callable[[ReIDSettings, DeviceManager], BaseEmbedder]] = {
    "osnet_ain_x1_0": OsNetEmbedder,
}


def create_embedder(config: ReIDSettings, device_manager: DeviceManager) -> BaseEmbedder:
    """Build the embedder named in configuration.

    Args:
        config: Validated ReID configuration.
        device_manager: Resolved compute device.

    Returns:
        A ready embedder. Call :meth:`BaseEmbedder.warmup` before the first frame.

    Raises:
        ConfigError: If ``reid.name`` is unknown.
        ModelLoadError: If the weights cannot be loaded.
    """
    try:
        factory = _EMBEDDERS[config.name.lower()]
    except KeyError:
        raise ConfigError(
            f"Unknown ReID model: {config.name!r}.",
            details={"name": config.name, "available": sorted(_EMBEDDERS)},
        ) from None
    return factory(config, device_manager)
