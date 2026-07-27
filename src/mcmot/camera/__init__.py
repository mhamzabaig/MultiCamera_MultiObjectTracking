"""Camera capture.

Webcams, RTSP cameras, HTTP/MJPEG streams (what a phone IP-camera app serves), and
video files all arrive through one interface, so nothing downstream knows or cares
which it is talking to.
"""

from __future__ import annotations

from mcmot.camera.capture import CaptureState, CaptureThread
from mcmot.camera.manager import CameraManager
from mcmot.camera.source import CameraSource, CameraSpec, SourceKind

__all__ = [
    "CameraManager",
    "CameraSource",
    "CameraSpec",
    "CaptureState",
    "CaptureThread",
    "SourceKind",
]
