"""Pipeline tests.

Fakes throughout: no GPU, no weights, no camera. The behaviours worth pinning down
here are the ones that fail *silently* in a real run — an identity quietly restarting
after a one-frame occlusion, a stalled camera still reporting 25 FPS, a single bad
frame killing a stream — because none of those raise, and all of them look fine on a
dashboard.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest

from mcmot.core.errors import CameraNotFoundError, PipelineError
from mcmot.core.settings import PipelineSettings
from mcmot.domain.detection import Detection
from mcmot.domain.frame import Frame, FrameMeta
from mcmot.domain.geometry import BBox
from mcmot.domain.identity import IdentityStatus, IdentityVerdict
from mcmot.domain.track import Track
from mcmot.pipeline import stream
from mcmot.pipeline.annotator import Annotator
from mcmot.pipeline.manager import PipelineManager
from mcmot.pipeline.stream import mjpeg_stream
from mcmot.pipeline.worker import PipelineState, PipelineWorker

CAMERA = "cam-1"
TRACK_BUFFER = 4  # small, so a retirement test does not need 30 frames


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def make_frame(index: int, *, age_s: float = 0.0) -> Frame:
    """A frame with a recognisable bright rectangle where a person would be."""
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[40:200, 80:180] = 200
    return Frame(
        image=image,
        meta=FrameMeta(
            camera_id=CAMERA,
            index=index,
            timestamp=time.time(),
            width=320,
            height=240,
            monotonic=time.monotonic() - age_s,
        ),
    )


class FakeCapture:
    """Hands out one scripted frame per call, then repeats the last one.

    Repeating rather than returning ``None`` is what a real capture thread does: the
    latest frame stays available until a newer one replaces it. The worker recognises
    it by index and idles.
    """

    def __init__(self, frames: list[Frame]) -> None:
        self.frames = frames
        self._next = 0

    def latest(self) -> Frame | None:
        if not self.frames:
            return None
        frame = self.frames[min(self._next, len(self.frames) - 1)]
        self._next += 1
        return frame


class FakeDetector:
    """Returns one detection per frame, or raises for the first ``fail`` calls."""

    def __init__(self, *, fail: int = 0) -> None:
        self.calls = 0
        self._fail = fail

    def detect(self, image: Any) -> list[Detection]:  # noqa: ARG002
        self.calls += 1
        if self.calls <= self._fail:
            raise RuntimeError("synthetic detector failure")
        return [Detection(bbox=BBox(80, 40, 180, 200), score=0.9)]

    def warmup(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-detector"}


class ScriptedTracker:
    """Returns the track ids listed for each successive frame.

    Scripting the tracker rather than the detector is what makes an occlusion
    expressible: ``[[1], [], [1]]`` is exactly one person missed for a single frame.
    """

    def __init__(self, script: list[list[int]]) -> None:
        self.script = script
        self.calls = 0

    def update(self, detections: list[Detection]) -> list[Track]:  # noqa: ARG002
        ids = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        return [
            Track(track_id=track_id, bbox=BBox(80, 40, 180, 200), score=0.9, hits=1)
            for track_id in ids
        ]

    def reset(self) -> None:
        self.calls = 0

    def describe(self) -> dict[str, Any]:
        return {"name": "scripted", "track_buffer": TRACK_BUFFER}


class FakeIdentity:
    """Records every call, and answers with a fixed verdict per track id."""

    def __init__(self, *, verdicts: dict[int, IdentityVerdict] | None = None) -> None:
        self.verdicts = verdicts or {}
        self.embed_calls: list[tuple[int, int]] = []
        self.retained: list[set[int]] = []
        self._embedded: set[int] = set()

    def should_embed(self, camera_id: str, track_id: int, frame_index: int) -> bool:  # noqa: ARG002
        # Once per track, so a test can tell "reused a cached verdict" apart from
        # "recomputed because the state was thrown away".
        return track_id not in self._embedded

    def identify(
        self, camera_id: str, track_id: int, crop: Any, *, frame_index: int = 0
    ) -> IdentityVerdict:  # noqa: ARG002
        self._embedded.add(track_id)
        self.embed_calls.append((track_id, frame_index))
        return self.verdicts.get(track_id, IdentityVerdict(status=IdentityStatus.UNKNOWN))

    def retain_tracks(self, camera_id: str, track_ids: set[int]) -> int:  # noqa: ARG002
        self.retained.append(set(track_ids))
        return 0


def build_worker(
    *,
    frames: list[Frame],
    script: list[list[int]],
    identity: FakeIdentity | None = None,
    detector: FakeDetector | None = None,
    config: PipelineSettings | None = None,
) -> tuple[PipelineWorker, FakeIdentity, ScriptedTracker, FakeDetector]:
    resolved_config = config or PipelineSettings()
    resolved_identity = identity or FakeIdentity()
    tracker = ScriptedTracker(script)
    resolved_detector = detector or FakeDetector()
    worker = PipelineWorker(
        camera_id=CAMERA,
        capture=FakeCapture(frames),  # type: ignore[arg-type]
        detector=resolved_detector,  # type: ignore[arg-type]
        tracker=tracker,  # type: ignore[arg-type]
        identity=resolved_identity,  # type: ignore[arg-type]
        annotator=Annotator(resolved_config),
        config=resolved_config,
        gpu_lock=threading.Lock(),
    )
    return worker, resolved_identity, tracker, resolved_detector


def run_until(predicate: Any, *, timeout: float = 5.0) -> bool:
    """Poll until ``predicate()`` is true. Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# Identity continuity — the reason _retire exists
# ---------------------------------------------------------------------------
def test_identity_survives_a_momentary_disappearance() -> None:
    """A track missing from one frame must keep its identity and its votes.

    ByteTrack only returns tracks it matched *this* frame, but keeps a lost one for
    ``track_buffer`` frames and revives it under the same id. Releasing state on
    absence would restart the vote for anyone who walks behind a chair — throwing
    away the hysteresis that stops the label flickering.
    """
    verdict = IdentityVerdict(
        status=IdentityStatus.IDENTIFIED, person_id="p1", name="Hamza", score=0.93, locked=True
    )
    identity = FakeIdentity(verdicts={1: verdict})
    # Present, present, gone, present.
    frames = [make_frame(i) for i in range(4)]
    worker, identity, tracker, _ = build_worker(
        frames=frames, script=[[1], [1], [], [1]], identity=identity
    )

    worker.start()
    assert run_until(lambda: tracker.calls >= 4)
    worker.stop()

    # One embedding only: the verdict cached on frame 0 was still there on frame 3.
    assert identity.embed_calls == [(1, 0)]
    # And the vote state was never released while the track could still come back.
    assert all(1 in retained for retained in identity.retained[:-1]), identity.retained


def test_track_is_retired_once_the_tracker_cannot_revive_it() -> None:
    """Past ``track_buffer`` frames of absence, state must actually be freed.

    The counterpart to the test above: a grace period that never expires is a leak on
    a camera that runs for days.
    """
    frames = [make_frame(i) for i in range(TRACK_BUFFER + 4)]
    script: list[list[int]] = [[1]] + [[] for _ in range(TRACK_BUFFER + 3)]
    worker, identity, tracker, _ = build_worker(frames=frames, script=script)

    worker.start()
    assert run_until(lambda: tracker.calls >= len(frames))
    worker.stop()

    # Retired at some point, and the release excluded the dead track.
    releases = [retained for retained in identity.retained if 1 not in retained]
    assert releases, identity.retained


# ---------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------
def test_stale_frames_are_skipped_not_processed() -> None:
    """A frame older than the limit costs inference time it can never repay."""
    config = PipelineSettings(drop_stale_frames=True, max_stale_age_s=0.2)
    stale = make_frame(0, age_s=1.0)
    fresh = make_frame(1)
    worker, _, tracker, detector = build_worker(
        frames=[stale, fresh], script=[[1], [1]], config=config
    )

    worker.start()
    assert run_until(lambda: worker.status()["frames_processed"] >= 1)
    worker.stop()

    status = worker.status()
    assert status["frames_skipped_stale"] == 1
    assert status["frames_processed"] == 1
    # Skipped means skipped: the detector never saw it.
    assert detector.calls == 1
    assert tracker.calls == 1


def test_publishes_a_decodable_annotated_jpeg() -> None:
    """The stream payload must be a real JPEG at the configured width."""
    config = PipelineSettings(stream_max_width=160, jpeg_quality=80)
    worker, _, _, _ = build_worker(frames=[make_frame(0)], script=[[1]], config=config)

    worker.start()
    assert run_until(lambda: worker.latest_jpeg() is not None)
    worker.stop()

    payload = worker.latest_jpeg()
    assert payload is not None
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape[1] == 160  # downscaled from 320


def test_frames_are_never_processed_twice() -> None:
    """The worker must recognise the latest frame it has already handled."""
    worker, _, tracker, _ = build_worker(frames=[make_frame(0)], script=[[1]])

    worker.start()
    assert run_until(lambda: tracker.calls >= 1)
    time.sleep(0.1)  # the capture keeps offering the same frame throughout
    worker.stop()

    assert tracker.calls == 1


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------
def test_a_transient_failure_does_not_kill_the_camera() -> None:
    """One bad frame must cost one frame, not the whole stream."""
    frames = [make_frame(i) for i in range(4)]
    worker, _, _, _ = build_worker(frames=frames, script=[[1]] * 4, detector=FakeDetector(fail=1))

    worker.start()
    assert run_until(lambda: worker.status()["frames_processed"] >= 2)
    worker.stop()

    assert worker.status()["state"] == PipelineState.STOPPED.value


def test_repeated_failure_fails_the_worker() -> None:
    """Sustained failure is a real fault and must surface, not retry forever."""
    frames = [make_frame(i) for i in range(20)]
    worker, _, _, _ = build_worker(frames=frames, script=[[1]] * 20, detector=FakeDetector(fail=20))

    worker.start()
    assert run_until(lambda: not worker.is_running)
    worker.stop()

    status = worker.status()
    assert status["state"] == PipelineState.FAILED.value
    assert "synthetic detector failure" in str(status["last_error"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_reported_fps_decays_when_the_camera_stalls() -> None:
    """A stalled pipeline must not keep reporting the rate it had while healthy.

    Measuring between the first and last processed frame freezes the figure forever,
    which is the reading most likely to be believed and most likely to be wrong.
    """
    worker, _, tracker, _ = build_worker(frames=[make_frame(i) for i in range(6)], script=[[1]] * 6)

    worker.start()
    assert run_until(lambda: tracker.calls >= 6)
    worker.stop()

    while_running = worker.status()["fps"]
    assert while_running > 0
    time.sleep(0.3)
    assert worker.status()["fps"] < while_running


def test_identified_never_exceeds_people() -> None:
    """Counts drawn from different sets would make the dashboard read as nonsense."""
    verdict = IdentityVerdict(status=IdentityStatus.IDENTIFIED, person_id="p1", name="Hamza")
    identity = FakeIdentity(verdicts={1: verdict, 2: verdict})
    # Two people, then only one: the other is cached but no longer present.
    worker, _, tracker, _ = build_worker(
        frames=[make_frame(i) for i in range(3)],
        script=[[1, 2], [1], [1]],
        identity=identity,
    )

    worker.start()
    assert run_until(lambda: tracker.calls >= 3)
    worker.stop()

    status = worker.status()
    assert status["identified"] <= status["people"]


# ---------------------------------------------------------------------------
# Annotator
# ---------------------------------------------------------------------------
def test_annotator_does_not_modify_the_source_frame() -> None:
    """Capture buffers are shared; drawing on one would corrupt other consumers."""
    annotator = Annotator(PipelineSettings())
    frame = make_frame(0)
    before = frame.image.copy()

    output = annotator.draw(
        frame.image, [Track(track_id=1, bbox=BBox(10, 10, 60, 150), score=0.9)], {}
    )

    assert np.array_equal(frame.image, before)
    assert not np.array_equal(output, before)


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (None, "Identifying..."),
        (IdentityVerdict(status=IdentityStatus.PENDING), "Identifying..."),
        (IdentityVerdict(status=IdentityStatus.UNKNOWN), "Unknown"),
        (IdentityVerdict(status=IdentityStatus.DISABLED), "No gallery"),
        (
            IdentityVerdict(status=IdentityStatus.IDENTIFIED, person_id="p1", name="Hamza"),
            "Hamza",
        ),
    ],
)
def test_label_distinguishes_undecided_from_stranger(
    verdict: IdentityVerdict | None, expected: str
) -> None:
    """Undecided and stranger are different claims, and must read differently."""
    assert Annotator._name_line(verdict) == expected


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class FakeCameraManager:
    def __init__(self, cameras: dict[str, FakeCapture]) -> None:
        self._cameras = cameras

    def get(self, camera_id: str) -> FakeCapture:
        try:
            return self._cameras[camera_id]
        except KeyError:
            raise CameraNotFoundError(f"no camera {camera_id!r}") from None


def build_manager(identity: FakeIdentity | None = None) -> PipelineManager:
    return PipelineManager(
        camera_manager=FakeCameraManager({CAMERA: FakeCapture([make_frame(0)])}),  # type: ignore[arg-type]
        detector=FakeDetector(),  # type: ignore[arg-type]
        identity=identity or FakeIdentity(),  # type: ignore[arg-type]
        tracker_factory=lambda: ScriptedTracker([[1]]),  # type: ignore[arg-type,return-value]
        config=PipelineSettings(),
    )


def test_manager_start_and_stop() -> None:
    manager = build_manager()
    manager.start(CAMERA)
    try:
        assert manager.is_running(CAMERA)
        assert manager.ids() == [CAMERA]
        assert manager.status(CAMERA)["camera_id"] == CAMERA
        assert len(manager.statuses()) == 1
    finally:
        manager.stop(CAMERA)

    assert not manager.is_running(CAMERA)
    assert manager.ids() == []


def test_manager_rejects_a_second_start() -> None:
    """Two workers on one camera would double GPU cost and fight over track ids."""
    manager = build_manager()
    manager.start(CAMERA)
    try:
        with pytest.raises(PipelineError):
            manager.start(CAMERA)
    finally:
        manager.stop(CAMERA)


def test_manager_rejects_an_unopened_camera() -> None:
    manager = build_manager()
    with pytest.raises(CameraNotFoundError):
        manager.start("nope")
    with pytest.raises(CameraNotFoundError):
        manager.status("nope")
    with pytest.raises(CameraNotFoundError):
        manager.stop("nope")


def test_stop_all_leaves_nothing_running() -> None:
    manager = build_manager()
    manager.start(CAMERA)
    manager.stop_all()
    assert manager.ids() == []
    assert not manager.is_running(CAMERA)


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------
async def test_stream_sends_each_new_frame_once() -> None:
    """Resending an unchanged buffer would saturate the link for no new picture."""
    frames = [b"jpeg-one", b"jpeg-two"]
    served: list[bytes | None] = [frames[0], frames[0], frames[1], None]

    def source() -> bytes | None:
        return served.pop(0) if served else None

    chunks = []
    async for chunk in mjpeg_stream(source, fps=1000, camera_id=CAMERA):
        chunks.append(chunk)
        if len(chunks) == 2:
            break

    assert b"jpeg-one" in chunks[0]
    assert b"jpeg-two" in chunks[1]
    assert b"Content-Type: image/jpeg" in chunks[0]


async def test_stream_ends_when_frames_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped pipeline must end the response, not hold every viewer open."""
    monkeypatch.setattr(stream, "_STALL_TIMEOUT_S", 0.05)
    frame = b"only-frame"

    chunks = [chunk async for chunk in mjpeg_stream(lambda: frame, fps=100, camera_id=CAMERA)]

    assert len(chunks) == 1


async def test_stream_gives_up_when_no_frame_ever_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A camera that never produces must not leave the request hanging."""
    monkeypatch.setattr(stream, "_STARTUP_GRACE_S", 0.05)

    chunks = [chunk async for chunk in mjpeg_stream(lambda: None, fps=100, camera_id=CAMERA)]

    assert chunks == []


async def test_stream_cancellation_propagates() -> None:
    """A closed tab is normal, and must not be swallowed into a hung task."""

    async def consume() -> None:
        async for _ in mjpeg_stream(lambda: b"frame", fps=10, camera_id=CAMERA):
            await asyncio.sleep(10)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_label_stays_inside_the_frame() -> None:
    """A person at the frame edge is exactly who most needs a readable label."""
    annotator = Annotator(PipelineSettings())
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    verdict = IdentityVerdict(status=IdentityStatus.IDENTIFIED, person_id="p1", name="Hamza")

    # Hard against the right edge, where an unclamped label would run off.
    annotator._draw_track(
        canvas, Track(track_id=1, bbox=BBox(300, 60, 320, 200), score=0.9), verdict
    )

    # Something was drawn in the rightmost columns, and nothing was lost off-canvas:
    # the text block starts far enough left to fit.
    painted = np.argwhere(canvas.any(axis=2))
    assert painted.size > 0
    assert painted[:, 1].min() < 300  # label pulled left of the box
