"""Cross-cutting concerns: configuration, logging, errors, device, composition.

This package is imported by every other module and imports from none of them.
That one-way dependency is what guarantees the package graph stays acyclic as the
system grows.
"""

from __future__ import annotations

from mcmot.core.container import Container
from mcmot.core.device import DeviceInfo, DeviceManager
from mcmot.core.errors import MCMOTError
from mcmot.core.logging import Throttle, bind_context, configure_logging, get_logger
from mcmot.core.settings import Settings, load_settings

__all__ = [
    "Container",
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
