"""Per-camera pipeline worker.

One thread per camera, running the whole chain:

    latest frame -> detect -> track -> (sample) identify -> annotate -> encode

Four decisions in here are what make it work in practice rather than only in a demo.

**It pulls, never subscribes.** The worker takes whatever frame the capture thread has
*now*. If it fell behind, the frames it missed are simply gone. Nothing queues, so a
slow moment costs coverage, never freshness — and coverage loss is counted.

**Identity is sampled, not recomputed.** Embedding costs ~21 ms, so running it per
person per frame is unaffordable. The Identity Service decides when a track is due
(every 10th frame, every 60th once locked) and the worker reuses that track's last
verdict in between. That is why the label on screen stays put instead of flickering.

**GPU work is serialised.** All workers share one lock around detection and embedding.
There is one GPU; letting two cameras interleave kernel launches would make latency
unpredictable and gains nothing. Serialising makes the budget legible.

**Stale frames are skipped, not processed.** If a frame is already older than
``max_stale_age_s`` by the time we reach it, running 35 ms of inference on it only
makes the next frame staler too. Dropping it is how the loop recovers.

**Identity outlives a momentary disappearance.** ByteTrack returns only the tracks it
matched *this* frame, but keeps a lost one alive internally for ``track_buffer``
frames and revives it under the same id. So a track missing from one frame is not
gone, and discarding its identity would restart the vote — throwing away the very
hysteresis phase 6 exists to provide. State is retired on the tracker's own horizon
instead of on per-frame presence.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import cv2

from mcmot.camera.capture import CaptureThread
from mcmot.core.logging import Throttle, bind_context, get_logger
from mcmot.core.settings import PipelineSettings
from mcmot.detection.base import BaseDetector
from mcmot.domain.frame import Frame
from mcmot.domain.identity import IdentityVerdict
from mcmot.domain.track import Track
from mcmot.identity.service import IdentityService
from mcmot.pipeline.annotator import Annotator
from mcmot.tracking.base import BaseTracker

__all__ = ["PipelineState", "PipelineWorker"]

_log = get_logger(__name__)

_LATENCY_WINDOW = 60
#: How long to wait for a new frame before looking again. Short enough not to add
#: perceptible latency, long enough not to spin a core on an idle camera.
_IDLE_POLL_S = 0.004
#: Consecutive failures tolerated before the worker gives up. A single frame can fail
#: for reasons that do not recur — a truncated JPEG, a transient CUDA error — and
#: killing the camera for one of those would be worse than skipping the frame.
_MAX_CONSECUTIVE_ERRORS = 5
#: Fallback retirement horizon when a tracker does not report ``track_buffer``.
_DEFAULT_TRACK_GRACE = 30


class PipelineState(str, Enum):
    """Lifecycle state of a pipeline worker."""

    IDLE = "idle"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(slots=True)
class _Stats:
    """Rolling counters and timings for one worker."""

    processed: int = 0
    skipped_stale: int = 0
    people: int = 0
    identified: int = 0
    embeddings: int = 0
    total_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    detect_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    reid_ms: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    finished: deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    """Completion times on the *performance* counter. Not ``time.monotonic``: its
    resolution on Windows is ~15 ms, so several frames land on the same tick and a
    short window can span exactly zero, reporting 0 FPS while running fine."""

    @staticmethod
    def _mean(values: deque[float]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of the current metrics."""
        fps = 0.0
        if len(self.finished) > 1:
            # Measured to *now*, not to the last frame. Dividing by the span between
            # the first and last processed frame would freeze the reading: a camera
            # that stalled would keep reporting the healthy rate it had before it
            # died. Ending the window at now makes the figure decay to zero instead.
            span = time.perf_counter() - self.finished[0]
            if span > 0:
                fps = round((len(self.finished) - 1) / span, 2)
        return {
            "fps": fps,
            "latency_ms": self._mean(self.total_ms),
            "detect_ms": self._mean(self.detect_ms),
            "reid_ms": self._mean(self.reid_ms),
            "people": self.people,
            "identified": self.identified,
            "frames_processed": self.processed,
            "frames_skipped_stale": self.skipped_stale,
            "embeddings_computed": self.embeddings,
        }


class PipelineWorker:
    """Runs detection, tracking, and identification for one camera."""

    def __init__(
        self,
        *,
        camera_id: str,
        capture: CaptureThread,
        detector: BaseDetector,
        tracker: BaseTracker,
        identity: IdentityService,
        annotator: Annotator,
        config: PipelineSettings,
        gpu_lock: threading.Lock,
    ) -> None:
        """Wire a worker.

        Args:
            camera_id: Camera this worker serves.
            capture: The camera's capture thread.
            detector: Shared person detector.
            tracker: This camera's own tracker (never shared).
            identity: Shared identity service.
            annotator: Frame annotator.
            config: Validated pipeline configuration.
            gpu_lock: Lock shared by all workers, serialising GPU inference.
        """
        self._camera_id = camera_id
        self._capture = capture
        self._detector = detector
        self._tracker = tracker
        self._identity = identity
        self._annotator = annotator
        self._config = config
        self._gpu_lock = gpu_lock

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._state = PipelineState.IDLE
        self._error: str | None = None
        self._stats = _Stats()
        self._jpeg: bytes | None = None
        self._last_frame_index = -1
        # Last verdict per track, so a label persists between sampled embeddings.
        self._verdicts: dict[int, IdentityVerdict] = {}
        # Processed-frame counter and the tick each track was last matched on. This
        # counter, not the camera's frame index, is the tracker's own clock: the
        # tracker advances once per update() call, so its track_buffer is measured in
        # processed frames. Comparing against camera indices would retire identities
        # early whenever frames are being skipped.
        self._tick = 0
        self._last_seen: dict[int, int] = {}
        self._track_grace = int(tracker.describe().get("track_buffer", _DEFAULT_TRACK_GRACE))
        self._metrics_throttle = Throttle(5.0)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        """Start processing in a background thread."""
        if self.is_running:
            return
        self._stop.clear()
        self._set_state(PipelineState.RUNNING)
        self._thread = threading.Thread(
            target=self._run, name=f"pipeline-{self._camera_id}", daemon=True
        )
        self._thread.start()
        _log.info("pipeline.started", camera_id=self._camera_id)

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the worker to stop and wait briefly for it."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self._thread = None
        if self._state is not PipelineState.FAILED:
            self._set_state(PipelineState.STOPPED)
        # Release this camera's vote state; the tracks are gone.
        self._identity.retain_tracks(self._camera_id, set())
        _log.info("pipeline.stopped", camera_id=self._camera_id, **self._stats.snapshot())

    @property
    def is_running(self) -> bool:
        """Whether the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    # -- output -----------------------------------------------------------
    def latest_jpeg(self) -> bytes | None:
        """Return the most recent annotated frame as JPEG bytes."""
        with self._lock:
            return self._jpeg

    def status(self) -> dict[str, Any]:
        """Return a serialisable snapshot for the API and the dashboard."""
        with self._lock:
            return {
                "camera_id": self._camera_id,
                "state": self._state.value,
                "running": self.is_running,
                "last_error": self._error,
                **self._stats.snapshot(),
            }

    # -- the loop ---------------------------------------------------------
    def _run(self) -> None:
        with bind_context(camera_id=self._camera_id):
            failures = 0
            while not self._stop.is_set():
                frame = self._next_frame()
                if frame is None:
                    continue
                try:
                    self._process(frame)
                except Exception as exc:  # noqa: BLE001 - one frame must not kill a camera
                    failures += 1
                    self._error = str(exc)
                    if failures >= _MAX_CONSECUTIVE_ERRORS:
                        self._set_state(PipelineState.FAILED)
                        _log.exception("pipeline.failed", camera_id=self._camera_id)
                        return
                    _log.warning(
                        "pipeline.frame_failed",
                        camera_id=self._camera_id,
                        consecutive=failures,
                        error=str(exc),
                    )
                else:
                    failures = 0

    def _next_frame(self) -> Frame | None:
        """Return a fresh unprocessed frame, or ``None`` to loop again."""
        frame = self._capture.latest()
        if frame is None or frame.meta.index == self._last_frame_index:
            # Nothing new. Wait briefly rather than spinning.
            self._stop.wait(_IDLE_POLL_S)
            return None

        if self._config.drop_stale_frames and frame.meta.age_s() > self._config.max_stale_age_s:
            # Already old. Processing it would only delay the next one further.
            self._last_frame_index = frame.meta.index
            with self._lock:
                self._stats.skipped_stale += 1
            return None

        self._last_frame_index = frame.meta.index
        return frame

    def _process(self, frame: Frame) -> None:
        """Run one frame all the way through and publish the result."""
        started = time.perf_counter()
        self._tick += 1

        with self._gpu_lock:
            detect_started = time.perf_counter()
            detections = self._detector.detect(frame.image)
            detect_ms = (time.perf_counter() - detect_started) * 1000

        tracks = self._tracker.update(detections)

        reid_started = time.perf_counter()
        embedded = self._resolve_identities(frame, tracks)
        reid_ms = (time.perf_counter() - reid_started) * 1000

        self._retire(tracks)

        if self._config.annotate:
            output = self._annotator.draw(frame.image, tracks, self._verdicts)
        else:
            output = frame.image
        self._publish(output)

        total_ms = (time.perf_counter() - started) * 1000
        self._record(tracks, detect_ms, reid_ms, total_ms, embedded)

    def _resolve_identities(self, frame: Frame, tracks: list[Track]) -> int:
        """Identify the tracks that are due; reuse cached verdicts for the rest.

        Returns:
            How many embeddings were computed for this frame.
        """
        embedded = 0
        height, width = frame.shape
        for track in tracks:
            if not self._identity.should_embed(self._camera_id, track.track_id, frame.meta.index):
                continue

            x1, y1, x2, y2 = track.bbox.clip(width, height).to_int()
            crop = frame.image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            with self._gpu_lock:
                verdict = self._identity.identify(
                    self._camera_id,
                    track.track_id,
                    crop,
                    frame_index=frame.meta.index,
                )
            embedded += 1

            previous = self._verdicts.get(track.track_id)
            self._verdicts[track.track_id] = verdict
            if previous is None or previous.person_id != verdict.person_id:
                # Log identity changes only. Logging every verdict would be one line
                # per person per sampled frame.
                _log.info(
                    "identity.changed",
                    track_id=track.track_id,
                    name=verdict.display_name,
                    status=verdict.status.value,
                    score=round(verdict.score, 3),
                    locked=verdict.locked,
                )
        return embedded

    def _retire(self, tracks: list[Track]) -> None:
        """Release state for tracks the tracker can no longer revive.

        A track absent from this frame is *not* necessarily gone — ByteTrack keeps a
        lost track for ``track_buffer`` frames and can match it again under the same
        id. Retiring on the same horizon means an occluded person keeps their name and
        their accumulated votes, while a track that truly left still frees its state
        rather than accumulating for the life of the process.
        """
        for track in tracks:
            self._last_seen[track.track_id] = self._tick

        expired = [
            track_id
            for track_id, tick in self._last_seen.items()
            if self._tick - tick > self._track_grace
        ]
        for track_id in expired:
            del self._last_seen[track_id]
            self._verdicts.pop(track_id, None)

        if expired:
            # Only worth a call when something actually left; otherwise this would
            # walk the whole vote table on every frame to delete nothing.
            self._identity.retain_tracks(self._camera_id, set(self._last_seen))

    def _publish(self, image: Any) -> None:
        """Downscale, JPEG-encode, and store the frame for streaming."""
        limit = self._config.stream_max_width
        if image.shape[1] > limit:
            # Encoding a 1280-wide frame costs several times more than a 960-wide
            # one, and the browser is displaying it small anyway.
            scale = limit / image.shape[1]
            image = cv2.resize(
                image,
                (limit, max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality]
        )
        if not ok:  # pragma: no cover - encoder failure is not reproducible here
            return
        with self._lock:
            self._jpeg = buffer.tobytes()

    def _record(
        self,
        tracks: list[Track],
        detect_ms: float,
        reid_ms: float,
        total_ms: float,
        embedded: int,
    ) -> None:
        # Counted over the tracks in *this* frame, not over the verdict cache: the
        # cache also holds tracks awaiting retirement, which would let `identified`
        # exceed `people` and make the dashboard read as nonsense.
        live = {track.track_id for track in tracks}
        with self._lock:
            stats = self._stats
            stats.processed += 1
            stats.people = len(tracks)
            stats.identified = sum(
                1
                for track_id, verdict in self._verdicts.items()
                if track_id in live and verdict.is_identified
            )
            stats.embeddings += embedded
            stats.detect_ms.append(detect_ms)
            stats.reid_ms.append(reid_ms)
            stats.total_ms.append(total_ms)
            stats.finished.append(time.perf_counter())
            snapshot = stats.snapshot()

        # Aggregates every few seconds, never per frame.
        if self._metrics_throttle.ready(self._camera_id):
            _log.info("pipeline.metrics", camera_id=self._camera_id, **snapshot)

    def _set_state(self, state: PipelineState) -> None:
        with self._lock:
            self._state = state
