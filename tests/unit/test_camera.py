"""Camera layer tests.

Focused on the counters and error types, because every defect found in this layer so
far has been a *reporting* defect rather than a functional one: capture kept working
while the numbers it published were wrong, which is the failure mode that survives
manual testing and then misleads whoever reads the dashboard.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from mcmot.camera.capture import CaptureThread
from mcmot.camera.manager import CameraManager
from mcmot.camera.source import CameraSpec, SourceKind
from mcmot.core.errors import CameraAlreadyOpenError, CameraNotFoundError, InvalidCameraSourceError
from mcmot.core.settings import Settings


@pytest.fixture
def clip(tmp_path: Path) -> str:
    """A short generated video, so no test needs a real camera."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 240))
    rng = np.random.default_rng(3)
    for _ in range(120):
        writer.write(rng.integers(0, 200, size=(240, 160, 3), dtype=np.uint8))
    writer.release()
    return str(path)


class TestSourceParsing:
    @pytest.mark.parametrize(
        ("source", "kind"),
        [
            ("0", SourceKind.WEBCAM),
            ("rtsp://192.0.2.1:554/live", SourceKind.RTSP),
            ("http://192.168.1.42:8080/video", SourceKind.HTTP),
        ],
    )
    def test_recognises_each_kind(self, source: str, kind: SourceKind) -> None:
        assert CameraSpec.parse("cam", source).kind is kind

    def test_recognises_a_file(self, clip: str) -> None:
        assert CameraSpec.parse("cam", clip).kind is SourceKind.FILE

    @pytest.mark.parametrize("source", ["", "   ", "not-a-camera", "ftp://host/x"])
    def test_rejects_nonsense_with_a_client_error(self, source: str) -> None:
        """422, not the base class's 502: the request was wrong, not the camera."""
        with pytest.raises(InvalidCameraSourceError) as caught:
            CameraSpec.parse("cam", source)

        assert int(caught.value.http_status) == 422

    def test_rejects_a_blank_id(self) -> None:
        with pytest.raises(InvalidCameraSourceError):
            CameraSpec.parse("  ", "0")


class TestCredentialRedaction:
    def test_strips_credentials_from_a_url(self) -> None:
        """An IP camera password must not reach a log, a response, or a screen."""
        spec = CameraSpec.parse("ip", "rtsp://admin:hunter2@192.0.2.9:554/stream")

        assert "hunter2" not in spec.redacted_source
        assert "192.0.2.9" in spec.redacted_source
        assert spec.to_dict()["source"] == spec.redacted_source

    def test_keeps_the_real_source_for_dialling(self) -> None:
        """Only the redacted form is published; the connection still needs the real one."""
        spec = CameraSpec.parse("ip", "rtsp://admin:hunter2@192.0.2.9:554/stream")

        assert spec.source == "rtsp://admin:hunter2@192.0.2.9:554/stream"

    def test_leaves_a_credential_free_url_alone(self) -> None:
        spec = CameraSpec.parse("ip", "http://192.168.1.42:8080/video")

        assert spec.redacted_source == "http://192.168.1.42:8080/video"


class TestCounters:
    def test_dropped_counts_only_frames_nobody_read(self, clip: str, settings: Settings) -> None:
        """The slot being full is not a drop if a consumer has already seen its frame.

        With a peeking consumer — which is what the pipeline is — counting "the slot
        was occupied" reports a ~100% drop rate on a perfectly healthy camera. A
        meaningless number is worse than none: it is the reading that makes a real
        problem look normal.
        """
        capture = CaptureThread(CameraSpec.parse("cam", clip), settings.camera)
        capture.start()
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                capture.latest()  # read every frame, as a fast consumer would
                time.sleep(0.005)
            status = capture.status()
        finally:
            capture.stop()

        assert status["frames_delivered"] > 5
        # A reader keeping up misses almost nothing; the old counter would have
        # reported a drop for all but the first frame.
        assert status["frames_dropped"] < status["frames_delivered"] / 2

    def test_dropped_counts_up_when_nobody_reads(self, clip: str, settings: Settings) -> None:
        """The counter must still work — it is how coverage loss stays visible."""
        capture = CaptureThread(CameraSpec.parse("cam", clip), settings.camera)
        capture.start()
        try:
            time.sleep(2.0)  # never read
            status = capture.status()
        finally:
            capture.stop()

        assert status["frames_dropped"] > 0

    def test_reported_fps_decays_once_frames_stop(self, clip: str, settings: Settings) -> None:
        """A dead camera must not keep reporting the rate it had while healthy.

        Measured between the first and last captured frame, the figure freezes, so a
        camera that left Wi-Fi shows `state=reconnecting` beside a confident 25 fps.
        """
        capture = CaptureThread(CameraSpec.parse("cam", clip), settings.camera)
        capture.start()
        try:
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                capture.latest()
                time.sleep(0.01)
            while_streaming = capture.status()["fps"]
            assert while_streaming > 0

            # The clip is ~4.8s long, so waiting past its end stops the frames.
            time.sleep(4.0)
            after = capture.status()["fps"]
        finally:
            capture.stop()

        assert after < while_streaming


class TestManager:
    def test_rejects_a_duplicate_id_as_a_conflict(self, clip: str, settings: Settings) -> None:
        manager = CameraManager(settings.camera)
        manager.open("cam", clip)
        try:
            with pytest.raises(CameraAlreadyOpenError) as caught:
                manager.open("cam", clip)
            assert int(caught.value.http_status) == 409
        finally:
            manager.close_all()

    def test_unknown_id_is_not_found(self, settings: Settings) -> None:
        manager = CameraManager(settings.camera)

        with pytest.raises(CameraNotFoundError):
            manager.status("nope")

    def test_leaves_no_entry_behind_after_a_failed_open(self, settings: Settings) -> None:
        manager = CameraManager(settings.camera)

        with pytest.raises(InvalidCameraSourceError):
            manager.open("cam", "not-a-camera")

        assert manager.ids() == []
