"""Capture thread: decouples reading frames from processing them.

The core policy is **latest-frame-wins**. Capture runs as fast as the source
delivers, and each new frame *replaces* the previous one whether or not anything
consumed it. Dropped frames are counted, not queued.

That is the opposite of the obvious design, and it is the whole point. If the
pipeline is slower than the camera — which it always is at some moment: a GPU hiccup,
a burst of people, another camera starting — then a queue grows without bound and
every frame the pipeline handles is older than the last. The system keeps reporting a
healthy frame rate while showing the past. For live tracking, a frame that arrived
while we were busy has no value: the person has already moved. Dropping it is correct.

The second job here is surviving disconnection. A phone leaves Wi-Fi range, a laptop
sleeps, a PoE switch reboots. The thread reconnects with exponential backoff and
keeps its frame counter monotonic, so a reconnect looks like a gap in frame indices
rather than a restart — which is exactly what downstream latency and identity logic
need in order to notice that time passed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from enum import Enum
from typing import Any

from mcmot.camera.source import CameraSource, CameraSpec
from mcmot.core.errors import CameraError
from mcmot.core.logging import Throttle, bind_context, get_logger
from mcmot.core.settings import CameraSettings
from mcmot.domain.frame import Frame

__all__ = ["CaptureState", "CaptureThread"]

_log = get_logger(__name__)

#: Window over which the delivered frame rate is averaged.
_FPS_WINDOW = 30


class CaptureState(str, Enum):
    """Lifecycle state of a capture thread."""

    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    """A file source reached its end. Not an error."""

    FAILED = "failed"
    """Gave up after exhausting ``max_reconnect_attempts``."""

    STOPPED = "stopped"


class CaptureThread:
    """Reads frames from one source in the background, keeping only the newest."""

    def __init__(self, spec: CameraSpec, config: CameraSettings) -> None:
        """Create a stopped capture thread.

        Args:
            spec: Resolved source specification.
            config: Validated camera configuration.
        """
        self._spec = spec
        self._config = config
        self._source = CameraSource(spec, config)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._latest: Frame | None = None
        self._state = CaptureState.IDLE
        self._error: str | None = None

        self._frame_index = 0
        self._delivered = 0
        self._dropped = 0
        self._reconnects = 0
        self._timestamps: deque[float] = deque(maxlen=_FPS_WINDOW)
        self._log_throttle = Throttle(5.0)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Start capturing.

        Returns immediately; the first frame arrives asynchronously. Connection
        failures surface through :meth:`status` rather than from here, because a
        camera that is merely not plugged in yet should not fail an API call.

        Raises:
            CameraError: If already running.
        """
        if self._thread is not None and self._thread.is_alive():
            raise CameraError(
                "Capture is already running.", details={"camera_id": self._spec.camera_id}
            )

        self._stop.clear()
        self._set_state(CaptureState.CONNECTING)
        # Daemon: a hung network read must never prevent the process from exiting.
        self._thread = threading.Thread(
            target=self._run, name=f"capture-{self._spec.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the thread to stop and wait briefly for it.

        Args:
            timeout: Seconds to wait for the thread to finish.
        """
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            # A read already in flight can outlast this; the source is released in
            # the thread's own finally block, and the thread is a daemon.
            thread.join(timeout=timeout)
        self._thread = None
        self._set_state(CaptureState.STOPPED)
        _log.info(
            "camera.stopped",
            camera_id=self._spec.camera_id,
            delivered=self._delivered,
            dropped=self._dropped,
            reconnects=self._reconnects,
        )

    @property
    def is_running(self) -> bool:
        """Whether the capture thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # -- frame access -----------------------------------------------------
    def latest(self) -> Frame | None:
        """Return the most recent frame, or ``None`` if none has arrived yet.

        Non-consuming: repeated calls return the same frame until a newer one
        arrives. The pipeline pairs this with the frame index to notice repeats.
        """
        with self._lock:
            return self._latest

    def take(self) -> Frame | None:
        """Return the most recent frame and clear it, so it is delivered once."""
        with self._lock:
            frame = self._latest
            self._latest = None
            return frame

    # -- status -----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Return a serialisable snapshot for the API and the dashboard."""
        with self._lock:
            latest = self._latest
            return {
                **self._spec.to_dict(),
                "state": self._state.value,
                "running": self.is_running,
                "fps": self._fps(),
                "frames_delivered": self._delivered,
                "frames_dropped": self._dropped,
                "reconnects": self._reconnects,
                "last_error": self._error,
                "frame_index": self._frame_index,
                "latency_ms": (round(latest.meta.age_s() * 1000, 1) if latest else None),
            }

    def _fps(self) -> float:
        """Delivered frame rate over the recent window. Caller holds the lock."""
        if len(self._timestamps) < 2:
            return 0.0
        span = self._timestamps[-1] - self._timestamps[0]
        return round((len(self._timestamps) - 1) / span, 2) if span > 0 else 0.0

    # -- the thread -------------------------------------------------------
    def _run(self) -> None:
        """Open, then read until stopped, reconnecting as needed."""
        with bind_context(camera_id=self._spec.camera_id):
            delay = self._config.reconnect_initial_delay_s
            attempts = 0
            try:
                while not self._stop.is_set():
                    try:
                        if not self._source.is_open:
                            self._set_state(
                                CaptureState.RECONNECTING if attempts else CaptureState.CONNECTING
                            )
                            self._source.open()
                            self._set_state(CaptureState.STREAMING)
                            self._error = None
                            # Reset the backoff only after a successful open, so a
                            # source that flaps does not hammer reconnects.
                            delay = self._config.reconnect_initial_delay_s
                            attempts = 0

                        started = time.monotonic()
                        self._publish(self._source.read())
                        self._pace(started)

                    except Exception as exc:  # noqa: BLE001 - any fault means reconnect
                        self._source.close()
                        self._error = str(exc)

                        if not self._spec.is_live:
                            # A file running out is a normal end, not a failure.
                            self._set_state(CaptureState.ENDED)
                            _log.info("camera.ended", frames=self._delivered)
                            return

                        attempts += 1
                        limit = self._config.max_reconnect_attempts
                        if limit and attempts > limit:
                            self._set_state(CaptureState.FAILED)
                            _log.error("camera.failed", attempts=attempts, last_error=self._error)
                            return

                        self._set_state(CaptureState.RECONNECTING)
                        if self._log_throttle.ready(self._spec.camera_id):
                            _log.warning(
                                "camera.reconnecting",
                                attempt=attempts,
                                delay_s=round(delay, 2),
                                error=self._error,
                            )
                        self._reconnects += 1
                        # Interruptible sleep: stop() must not wait out the backoff.
                        if self._stop.wait(delay):
                            break
                        delay = min(delay * 2, self._config.reconnect_max_delay_s)
            finally:
                self._source.close()
                if self._state not in (CaptureState.ENDED, CaptureState.FAILED):
                    self._set_state(CaptureState.STOPPED)

    def _pace(self, frame_started: float) -> None:
        """Throttle a file source to real time. A no-op for live sources.

        A live source paces itself: ``read()`` blocks until the camera produces the
        next frame. A file does not — it was measured returning 1800 FPS, burning a
        core and consuming a six-second clip in under a second. That makes a video
        file useless as a stand-in for a camera, both for testing the pipeline and
        for the legitimate case of pointing the tracking page at a recording.

        Uses the file's own encoded rate, falling back to ``camera.target_fps``.
        """
        if self._spec.is_live:
            return
        fps = self._source.native_fps or float(self._config.target_fps)
        remaining = (1.0 / fps) - (time.monotonic() - frame_started)
        if remaining > 0:
            # Interruptible, so stop() does not have to wait out a frame interval.
            self._stop.wait(remaining)

    def _publish(self, image: Any) -> None:
        """Store a frame as the newest, counting any it replaces as dropped."""
        self._frame_index += 1
        frame = Frame.create(image, camera_id=self._spec.camera_id, index=self._frame_index)
        with self._lock:
            if self._latest is not None:
                # Nothing consumed the previous frame. Expected and healthy whenever
                # capture outpaces the pipeline; the count is what makes it visible.
                self._dropped += 1
            self._latest = frame
            self._delivered += 1
            self._timestamps.append(frame.meta.monotonic)

    def _set_state(self, state: CaptureState) -> None:
        with self._lock:
            self._state = state
