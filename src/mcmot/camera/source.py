"""Camera and video sources.

One class covers all four kinds — laptop webcam, RTSP camera, HTTP/MJPEG stream
(which is what a phone running an IP-camera app serves), and a video file — because
OpenCV opens all of them through ``VideoCapture``. The differences are which backend
to ask for and which properties to set, which is a handful of branches rather than a
class hierarchy.

Three settings here exist specifically to stop latency accumulating, the failure that
makes a naive implementation report 25 FPS while showing five-second-old frames:

``CAP_PROP_BUFFERSIZE = 1``
    Ask the driver to keep one frame, not a queue. A deeper buffer means every
    ``read()`` returns something older than what the camera is currently seeing.
``rtsp_transport=tcp``
    UDP silently drops packets under load, producing corrupt frames and stalls. TCP
    costs a little latency and returns whole frames.
``open/read timeouts as construction parameters``
    Without them a dead network stream blocks for OpenCV's 30 second default, so the
    capture thread sits there instead of reconnecting. Measurement showed these are
    honoured *only* when passed to the ``VideoCapture`` constructor — setting the
    properties afterwards, or using the FFMPEG env options, has no effect at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import cv2

from mcmot.core.errors import CameraDisconnectedError, CameraError
from mcmot.core.logging import get_logger
from mcmot.core.settings import CameraSettings
from mcmot.domain.types import ImageArray

__all__ = ["CameraSource", "CameraSpec", "SourceKind"]

_log = get_logger(__name__)


class SourceKind(str, Enum):
    """What kind of source a camera string refers to."""

    WEBCAM = "webcam"
    RTSP = "rtsp"
    HTTP = "http"
    """An HTTP stream, typically MJPEG — what phone IP-camera apps serve."""

    FILE = "file"
    """A video file. Ends rather than reconnects when it runs out."""


@dataclass(frozen=True, slots=True)
class CameraSpec:
    """A camera identifier plus its resolved source."""

    camera_id: str
    source: str
    kind: SourceKind

    @classmethod
    def parse(cls, camera_id: str, source: str | int) -> CameraSpec:
        """Work out what kind of source this is.

        Args:
            camera_id: Stable identifier for this camera.
            source: A device index (``0``, ``"0"``), a URL, or a file path.

        Returns:
            The resolved specification.

        Raises:
            CameraError: If the id is blank, or the source is empty or unrecognisable.
        """
        if not camera_id.strip():
            raise CameraError("Camera id must not be empty.")

        text = str(source).strip()
        if not text:
            raise CameraError("Camera source must not be empty.", details={"camera_id": camera_id})

        lowered = text.lower()
        if text.isdigit():
            kind = SourceKind.WEBCAM
        elif lowered.startswith("rtsp://"):
            kind = SourceKind.RTSP
        elif lowered.startswith(("http://", "https://")):
            kind = SourceKind.HTTP
        elif Path(text).is_file():
            kind = SourceKind.FILE
        else:
            raise CameraError(
                "Unrecognised camera source. Expected a device index (0), an "
                "rtsp:// or http:// URL, or a path to an existing video file.",
                details={"camera_id": camera_id, "source": text},
            )
        return cls(camera_id=camera_id.strip(), source=text, kind=kind)

    @property
    def is_live(self) -> bool:
        """Whether this source is a live stream, and so worth reconnecting to."""
        return self.kind is not SourceKind.FILE

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for API responses."""
        return {"camera_id": self.camera_id, "source": self.source, "kind": self.kind.value}


class CameraSource:
    """A single opened video source.

    Not thread-safe: exactly one :class:`~mcmot.camera.capture.CaptureThread` owns an
    instance and is the only caller.
    """

    def __init__(self, spec: CameraSpec, config: CameraSettings) -> None:
        """Create an unopened source.

        Args:
            spec: Resolved source specification.
            config: Validated camera configuration.
        """
        self._spec = spec
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    # -- properties -------------------------------------------------------
    @property
    def spec(self) -> CameraSpec:
        """The source specification."""
        return self._spec

    @property
    def is_open(self) -> bool:
        """Whether the source is currently open."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def native_fps(self) -> float:
        """The source's own frame rate, or ``0.0`` if it does not report one.

        Meaningful for files, which declare the rate they were encoded at. Live
        sources often report 0 or a nominal value, and it does not matter for them:
        a live ``read()`` blocks until the next frame, so the camera sets the pace.
        """
        if self._capture is None:
            return 0.0
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        return fps if 0.0 < fps < 1000.0 else 0.0

    # -- lifecycle --------------------------------------------------------
    def open(self) -> None:
        """Open the source and apply capture properties.

        Raises:
            CameraError: If the source cannot be opened.
        """
        self.close()

        if self._spec.kind is SourceKind.RTSP:
            # The transport choice is only settable through this env var, and FFMPEG
            # reads it while opening, so it must be set before constructing.
            # Timeouts go through construction parameters instead -- see _construct.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self._config.rtsp_transport}"
            )

        capture = self._construct()
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera source: {self._spec.source}",
                details=self._spec.to_dict(),
            )

        self._capture = capture
        self._apply_properties(capture)
        _log.info("camera.opened", **self._spec.to_dict(), **self._actual_properties())

    def _construct(self) -> cv2.VideoCapture:
        """Build the VideoCapture with the right backend for this source kind."""
        if self._spec.kind is SourceKind.WEBCAM:
            # DirectShow on Windows: the default MSMF backend takes seconds to open
            # a webcam and often ignores resolution requests.
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            return cv2.VideoCapture(int(self._spec.source), backend)
        if self._spec.kind is SourceKind.FILE:
            return cv2.VideoCapture(self._spec.source)

        # Network streams: timeouts must be passed as construction parameters, which
        # is the only form that takes effect. Measured against a dead RTSP address:
        #   no options                     30.02 s
        #   env OPENCV_FFMPEG timeout      30.03 s   (no effect)
        #   env OPENCV_FFMPEG stimeout     30.02 s   (no effect)
        #   params at construction          1.09 s   <- honoured
        # Without this a phone that leaves Wi-Fi blocks for 30 s per retry instead of
        # the configured timeout, so reconnection is effectively broken.
        return cv2.VideoCapture(
            self._spec.source,
            cv2.CAP_FFMPEG,
            [
                int(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC),
                int(self._config.open_timeout_s * 1000),
                int(cv2.CAP_PROP_READ_TIMEOUT_MSEC),
                int(self._config.read_timeout_s * 1000),
            ],
        )

    def _apply_properties(self, capture: cv2.VideoCapture) -> None:
        """Request buffer depth, timeouts, and (for webcams) resolution and rate."""
        # A one-frame buffer is the single most important setting for live latency.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self._config.buffer_size)

        if self._spec.kind is SourceKind.WEBCAM:
            # Requests, not guarantees: a webcam substitutes its nearest supported
            # mode, which is why the actual values are logged after opening.
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            capture.set(cv2.CAP_PROP_FPS, self._config.target_fps)

    def read(self) -> ImageArray:
        """Grab the next frame.

        Returns:
            A BGR ``uint8`` frame.

        Raises:
            CameraDisconnectedError: If the source is closed, has ended, or the read
                failed. The caller decides whether that means reconnect or stop,
                since a file ending is normal and a stream ending is not.
        """
        if self._capture is None:
            raise CameraDisconnectedError("Camera is not open.", details=self._spec.to_dict())

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraDisconnectedError(
                "Failed to read a frame from the camera.", details=self._spec.to_dict()
            )
        # OpenCV's stubs type this loosely, but every source here decodes to BGR
        # uint8. Cast rather than convert, since Frame validates dtype and shape at
        # the pipeline boundary anyway — a violation is caught there, not hidden.
        return cast("ImageArray", frame)

    def close(self) -> None:
        """Release the source. Safe to call repeatedly."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # -- introspection ----------------------------------------------------
    def _actual_properties(self) -> dict[str, Any]:
        """Return what the source actually negotiated, which may differ from config."""
        if self._capture is None:
            return {}
        return {
            "width": int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "source_fps": round(float(self._capture.get(cv2.CAP_PROP_FPS)), 2),
        }

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the source."""
        return {**self._spec.to_dict(), "open": self.is_open, **self._actual_properties()}

    def __enter__(self) -> CameraSource:
        """Open the source on context entry."""
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Release the source on context exit."""
        self.close()
