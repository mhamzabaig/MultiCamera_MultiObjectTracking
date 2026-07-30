"""Camera registry.

Owns the set of open cameras and their capture threads, so nothing above has to track
thread lifetimes. One entry per ``camera_id``; opening the same id twice is an error
rather than a silent replacement, because silently dropping a running stream is
exactly the kind of thing an operator does not notice until footage is missing.

Cameras live in memory only. They are not persisted, so a restart starts with none
open — the Camera Management page in phase 10 will re-open them from whatever the API
is told. There is no camera table yet because nothing needs one.
"""

from __future__ import annotations

import threading
from typing import Any

from mcmot.camera.capture import CaptureThread
from mcmot.camera.source import CameraSpec
from mcmot.core.errors import CameraAlreadyOpenError, CameraNotFoundError
from mcmot.core.logging import get_logger
from mcmot.core.settings import CameraSettings
from mcmot.domain.frame import Frame

__all__ = ["CameraManager"]

_log = get_logger(__name__)


class CameraManager:
    """Opens, tracks, and closes camera capture threads."""

    def __init__(self, config: CameraSettings) -> None:
        """Args: config: Validated camera configuration."""
        self._config = config
        self._cameras: dict[str, CaptureThread] = {}
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    def open(self, camera_id: str, source: str | int) -> CameraSpec:
        """Open a camera and start capturing.

        Returns as soon as the thread starts; the stream may still be connecting.
        Whether it succeeded is visible through :meth:`status`, so a camera that is
        simply not powered on yet does not fail the request that added it.

        Args:
            camera_id: Stable identifier, unique among open cameras.
            source: Device index, ``rtsp://``/``http://`` URL, or video file path.

        Returns:
            The resolved specification.

        Raises:
            InvalidCameraSourceError: If the source is unrecognisable.
            CameraAlreadyOpenError: If a camera is already open under this id.
        """
        spec = CameraSpec.parse(camera_id, source)
        with self._lock:
            if spec.camera_id in self._cameras:
                raise CameraAlreadyOpenError(
                    f"Camera {spec.camera_id!r} is already open. Close it first.",
                    details={"camera_id": spec.camera_id},
                )
            capture = CaptureThread(spec, self._config)
            self._cameras[spec.camera_id] = capture

        try:
            capture.start()
        except Exception:
            # Never leave an unstarted entry behind; the id must stay reusable.
            with self._lock:
                self._cameras.pop(spec.camera_id, None)
            raise

        _log.info("camera.registered", **spec.to_dict())
        return spec

    def close(self, camera_id: str) -> None:
        """Stop and remove a camera.

        Raises:
            CameraNotFoundError: If no camera is open under this id.
        """
        with self._lock:
            capture = self._cameras.pop(camera_id, None)
        if capture is None:
            raise CameraNotFoundError(details={"camera_id": camera_id})
        capture.stop()

    def close_all(self) -> None:
        """Stop every camera. Used at shutdown; never raises."""
        with self._lock:
            cameras = list(self._cameras.items())
            self._cameras.clear()
        for camera_id, capture in cameras:
            try:
                capture.stop()
            except Exception:  # noqa: BLE001 - shutdown must continue regardless
                _log.exception("camera.stop_failed", camera_id=camera_id)

    # -- access -----------------------------------------------------------
    def get(self, camera_id: str) -> CaptureThread:
        """Return a camera's capture thread.

        Raises:
            CameraNotFoundError: If no camera is open under this id.
        """
        with self._lock:
            capture = self._cameras.get(camera_id)
        if capture is None:
            raise CameraNotFoundError(details={"camera_id": camera_id})
        return capture

    def latest(self, camera_id: str) -> Frame | None:
        """Return a camera's most recent frame, or ``None`` if none has arrived.

        Raises:
            CameraNotFoundError: If no camera is open under this id.
        """
        return self.get(camera_id).latest()

    def is_open(self, camera_id: str) -> bool:
        """Whether a camera is registered under this id."""
        with self._lock:
            return camera_id in self._cameras

    def ids(self) -> list[str]:
        """Return the ids of every open camera."""
        with self._lock:
            return sorted(self._cameras)

    # -- status -----------------------------------------------------------
    def status(self, camera_id: str) -> dict[str, Any]:
        """Return one camera's status.

        Raises:
            CameraNotFoundError: If no camera is open under this id.
        """
        return self.get(camera_id).status()

    def statuses(self) -> list[dict[str, Any]]:
        """Return the status of every open camera."""
        with self._lock:
            cameras = list(self._cameras.values())
        return [capture.status() for capture in cameras]
