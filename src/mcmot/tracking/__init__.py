"""Multi-object tracking.

Same shape as :mod:`mcmot.detection`: a dict lookup, not a plugin registry. Adding
BoT-SORT or OC-SORT later means one new module and one entry in ``_TRACKERS``.
"""

from __future__ import annotations

from collections.abc import Callable

from mcmot.core.errors import ConfigError
from mcmot.core.settings import TrackerSettings
from mcmot.tracking.base import BaseTracker
from mcmot.tracking.bytetrack import ByteTrackTracker

__all__ = ["BaseTracker", "ByteTrackTracker", "create_tracker"]

#: Configured ``tracker.name`` -> factory.
_TRACKERS: dict[str, Callable[[TrackerSettings], BaseTracker]] = {
    "bytetrack": ByteTrackTracker,
}


def create_tracker(config: TrackerSettings) -> BaseTracker:
    """Build a tracker for one camera.

    Call once per camera, never shared: trackers hold per-track Kalman state.

    Args:
        config: Validated tracker configuration.

    Returns:
        A fresh tracker with no track history.

    Raises:
        ConfigError: If ``tracker.name`` is unknown.
    """
    try:
        factory = _TRACKERS[config.name.lower()]
    except KeyError:
        raise ConfigError(
            f"Unknown tracker: {config.name!r}.",
            details={"name": config.name, "available": sorted(_TRACKERS)},
        ) from None
    return factory(config)
