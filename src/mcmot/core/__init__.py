"""Cross-cutting concerns: configuration, logging, errors, device.

This package is imported by every other module and imports from none of them. That
one-way dependency is what guarantees the package graph stays acyclic as the system
grows, and it is why the composition root lives in :mod:`mcmot.container` rather than
here — wiring the feature packages together would make ``core`` depend on them.
"""

from __future__ import annotations

from mcmot.core.device import DeviceInfo, DeviceManager
from mcmot.core.errors import MCMOTError
from mcmot.core.logging import Throttle, bind_context, configure_logging, get_logger
from mcmot.core.settings import Settings, load_settings

__all__ = [
    "DeviceInfo",
    "DeviceManager",
    "MCMOTError",
    "Settings",
    "Throttle",
    "bind_context",
    "configure_logging",
    "get_logger",
    "load_settings",
]
