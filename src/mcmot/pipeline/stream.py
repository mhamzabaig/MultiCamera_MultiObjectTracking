"""MJPEG streaming.

``multipart/x-mixed-replace`` is the oldest trick on the web: send JPEG after JPEG on
one open response and the browser replaces the image in place. Chosen over WebRTC
because it is roughly twenty lines, needs no signalling, no ICE, and no codec
negotiation, and works in a plain ``<img src="...">`` on every browser including an
Android phone. It costs more bandwidth and around 100 ms more latency than WebRTC —
acceptable on a LAN, and the reason this lives behind a small interface.

The generator is async so that streaming to several viewers does not consume a thread
each: it awaits between frames rather than sleeping, leaving the event loop free.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable

from mcmot.core.logging import get_logger

__all__ = ["MJPEG_CONTENT_TYPE", "mjpeg_stream"]

_BOUNDARY = "mcmotframe"
MJPEG_CONTENT_TYPE = f"multipart/x-mixed-replace; boundary={_BOUNDARY}"

_log = get_logger(__name__)

#: How long to keep waiting for a first frame before giving up on the stream.
_STARTUP_GRACE_S = 15.0
#: How long to keep an established stream open with nothing new arriving. Without
#: this, stopping a pipeline would leave every viewer's response open forever,
#: resending the same final JPEG: the browser shows a frozen image and the request
#: never completes. Ending the response lets the page see the stream has finished.
_STALL_TIMEOUT_S = 5.0


async def mjpeg_stream(
    source: Callable[[], bytes | None],
    *,
    fps: int,
    camera_id: str = "",
) -> AsyncIterator[bytes]:
    """Yield MJPEG multipart chunks from a frame source.

    Args:
        source: Returns the newest JPEG bytes, or ``None`` if none is available.
            Called repeatedly; it must be cheap and non-blocking.
        fps: Target delivery rate. The cap matters: without it this would resend the
            same JPEG thousands of times a second and saturate the link.
        camera_id: Included in log lines.

    Yields:
        Multipart chunks, each a complete JPEG with its headers.
    """
    interval = 1.0 / max(fps, 1)
    previous: bytes | None = None
    delivered = 0
    # Measured on the clock, not counted in loop iterations: `asyncio.sleep` overshoots
    # (Windows' timer granularity is ~15 ms), so counting intervals would make a
    # "5 second" timeout mean anything from 5 to 75 seconds depending on `fps`.
    last_new = time.monotonic()

    try:
        while True:
            frame = source()

            # Identity comparison, not equality: the worker replaces the buffer
            # wholesale, so a new object *is* a new frame — and comparing megabytes of
            # JPEG per iteration would cost more than sending it.
            if frame is not None and frame is not previous:
                previous = frame
                delivered += 1
                last_new = time.monotonic()
                yield (
                    b"--" + _BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
                )
            else:
                # A camera still connecting is normal, so wait rather than closing
                # immediately — but not indefinitely once frames have stopped.
                idle = time.monotonic() - last_new
                limit = _STARTUP_GRACE_S if delivered == 0 else _STALL_TIMEOUT_S
                if idle >= limit:
                    _log.warning(
                        "stream.ended",
                        camera_id=camera_id,
                        delivered=delivered,
                        idle_s=round(idle, 1),
                    )
                    return

            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        # The viewer closed the tab or navigated away. Entirely normal.
        _log.debug("stream.cancelled", camera_id=camera_id, delivered=delivered)
        raise
