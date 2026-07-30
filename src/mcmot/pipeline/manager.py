"""Pipeline registry.

Owns one worker per camera and the single lock they share for GPU inference.

Collaborators are injected rather than pulled from the container, because the
container imports this module to build the manager — reaching back would be a cycle,
which is exactly the mistake corrected in phase 5.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from mcmot.camera.manager import CameraManager
from mcmot.core.errors import CameraNotFoundError, PipelineError
from mcmot.core.logging import get_logger
from mcmot.core.settings import PipelineSettings
from mcmot.detection.base import BaseDetector
from mcmot.identity.service import IdentityService
from mcmot.pipeline.annotator import Annotator
from mcmot.pipeline.worker import PipelineWorker
from mcmot.tracking.base import BaseTracker

__all__ = ["PipelineManager"]

_log = get_logger(__name__)


class PipelineManager:
    """Starts, tracks, and stops one pipeline worker per camera."""

    def __init__(
        self,
        *,
        camera_manager: CameraManager,
        detector: BaseDetector,
        identity: IdentityService,
        tracker_factory: Callable[[], BaseTracker],
        config: PipelineSettings,
    ) -> None:
        """Wire the manager.

        Args:
            camera_manager: Source of capture threads.
            detector: Shared person detector.
            identity: Shared identity service.
            tracker_factory: Builds a fresh tracker per camera, since trackers hold
                per-track Kalman state and must never be shared.
            config: Validated pipeline configuration.
        """
        self._cameras = camera_manager
        self._detector = detector
        self._identity = identity
        self._tracker_factory = tracker_factory
        self._config = config
        self._annotator = Annotator(config)

        self._workers: dict[str, PipelineWorker] = {}
        self._lock = threading.Lock()
        # One GPU, one lock, shared by every worker. Created here because this is the
        # only object that knows how many workers there are.
        self._gpu_lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------
    def start(self, camera_id: str) -> None:
        """Start tracking on an open camera.

        Args:
            camera_id: A camera already opened through the camera manager.

        Raises:
            CameraNotFoundError: If the camera is not open.
            PipelineError: If tracking is already running for it.
        """
        capture = self._cameras.get(camera_id)  # raises if the camera is not open

        with self._lock:
            existing = self._workers.get(camera_id)
            if existing is not None and existing.is_running:
                raise PipelineError(
                    f"Tracking is already running for camera {camera_id!r}.",
                    details={"camera_id": camera_id},
                )
            worker = PipelineWorker(
                camera_id=camera_id,
                capture=capture,
                detector=self._detector,
                tracker=self._tracker_factory(),
                identity=self._identity,
                annotator=self._annotator,
                config=self._config,
                gpu_lock=self._gpu_lock,
            )
            self._workers[camera_id] = worker

        worker.start()

    def stop(self, camera_id: str) -> None:
        """Stop tracking on a camera.

        Raises:
            CameraNotFoundError: If no pipeline exists for this camera.
        """
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is None:
            raise CameraNotFoundError(
                f"No tracking pipeline for camera {camera_id!r}.",
                details={"camera_id": camera_id},
            )
        worker.stop()

    def stop_all(self) -> None:
        """Stop every pipeline. Used at shutdown; never raises."""
        with self._lock:
            workers = list(self._workers.items())
            self._workers.clear()
        for camera_id, worker in workers:
            try:
                worker.stop()
            except Exception:  # noqa: BLE001 - shutdown must continue regardless
                _log.exception("pipeline.stop_failed", camera_id=camera_id)

    # -- access -----------------------------------------------------------
    def is_running(self, camera_id: str) -> bool:
        """Whether a pipeline is running for this camera."""
        with self._lock:
            worker = self._workers.get(camera_id)
        return worker is not None and worker.is_running

    def latest_jpeg(self, camera_id: str) -> bytes | None:
        """Return the newest annotated JPEG for a camera.

        Raises:
            CameraNotFoundError: If no pipeline exists for this camera.
        """
        return self._worker(camera_id).latest_jpeg()

    def frame_source(self, camera_id: str) -> Callable[[], bytes | None]:
        """Return a callable yielding the newest JPEG, for the MJPEG stream.

        Resolved once so the stream does not look the worker up per frame, and so a
        pipeline restarted mid-stream does not silently swap the viewer's source.

        Raises:
            CameraNotFoundError: If no pipeline exists for this camera.
        """
        return self._worker(camera_id).latest_jpeg

    def ids(self) -> list[str]:
        """Return the ids of every camera with a pipeline."""
        with self._lock:
            return sorted(self._workers)

    # -- status -----------------------------------------------------------
    def status(self, camera_id: str) -> dict[str, Any]:
        """Return one pipeline's status.

        Raises:
            CameraNotFoundError: If no pipeline exists for this camera.
        """
        return self._worker(camera_id).status()

    def statuses(self) -> list[dict[str, Any]]:
        """Return the status of every pipeline."""
        with self._lock:
            workers = list(self._workers.values())
        return [worker.status() for worker in workers]

    # -- helpers ----------------------------------------------------------
    def _worker(self, camera_id: str) -> PipelineWorker:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker is None:
            raise CameraNotFoundError(
                f"No tracking pipeline for camera {camera_id!r}.",
                details={"camera_id": camera_id},
            )
        return worker
