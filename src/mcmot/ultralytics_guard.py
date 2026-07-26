"""Guarded import for Ultralytics, shared by the detector and the tracker.

Ultralytics ships a feature called AutoUpdate: when it finds a requirement
missing, it **runs pip at runtime** and installs it. Observed here installing both
``lap`` and numpy 2.2.6 — the latter violating this project's ``numpy<2.2`` pin —
into whichever site-packages it could write to.

That is unacceptable in a system meant to run unattended:

* it mutates the environment mid-process, so a deployment's dependency set is no
  longer the one that was tested;
* it needs network access at inference time and fails confusingly without it;
* the freshly installed package is not importable until the process restarts, so
  the "fix" still crashes the current run — which is exactly what happened here.

``YOLO_AUTOINSTALL=false`` disables it, turning a silent environment mutation into
an honest error that names the missing dependency. Every Ultralytics import in this
codebase goes through :func:`import_ultralytics`.
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any

from mcmot.core.errors import ModelLoadError

__all__ = ["import_ultralytics"]


def import_ultralytics(module: str) -> Any:
    """Import an Ultralytics module with runtime auto-installation disabled.

    Args:
        module: Dotted module path, e.g. ``"ultralytics.trackers.byte_tracker"``.

    Returns:
        The imported module.

    Raises:
        ModelLoadError: If the module or one of its dependencies is missing, with
            the remedy named rather than silently attempted.
    """
    # Set before the import: Ultralytics reads these when its checks module loads.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    # Keeps its telemetry and version-check HTTP calls out of a local pipeline.
    os.environ.setdefault("YOLO_OFFLINE", "true")

    try:
        return import_module(module)
    except ImportError as exc:
        raise ModelLoadError(
            f"Could not import {module}: {exc}. "
            'Install the ml extra with `pip install -e ".[ml]"`.',
            details={"module": module, "reason": str(exc)},
        ) from exc
