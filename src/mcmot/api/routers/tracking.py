"""Tracking endpoints: start, stop, status, and the live video stream.

The stream is ``multipart/x-mixed-replace`` — MJPEG — which a browser renders in a
plain ``<img src="...">`` with no JavaScript, no signalling, and no codec
negotiation. It costs more bandwidth and roughly 100 ms more latency than WebRTC,
which is an acceptable trade on a LAN and is why the generator sits behind a small
interface that a WebRTC sink could replace.

The frame source is resolved **once**, when the response starts. A viewer therefore
keeps streaming from the worker it connected to; restarting the pipeline mid-stream
ends that response rather than silently swapping the picture for a different one.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Response, status
from fastapi.responses import StreamingResponse

from mcmot.api.deps import ContainerDep
from mcmot.api.schemas.system import ErrorResponse
from mcmot.api.schemas.tracking import TrackingResponse, TrackingStatus
from mcmot.core.logging import get_logger
from mcmot.pipeline import MJPEG_CONTENT_TYPE, mjpeg_stream

__all__ = ["router"]

_log = get_logger(__name__)

CameraId = Annotated[str, Path(description="Camera identifier.")]

router = APIRouter(
    prefix="/api/tracking",
    tags=["tracking"],
    responses={
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "Camera is not open, or no pipeline exists for it.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.get("", response_model=TrackingResponse, summary="List running pipelines")
def list_tracking(container: ContainerDep) -> TrackingResponse:
    """Return the status of every tracking pipeline."""
    return TrackingResponse(
        pipelines=[
            TrackingStatus.model_validate(status_dict)
            for status_dict in container.pipeline_manager.statuses()
        ]
    )


@router.post(
    "/{camera_id}/start",
    response_model=TrackingStatus,
    summary="Start tracking on an open camera",
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
            "description": "Tracking is already running for this camera.",
        }
    },
)
def start_tracking(container: ContainerDep, camera_id: CameraId) -> TrackingStatus:
    """Start the detect → track → identify pipeline for a camera.

    The first call in a process is the expensive one: touching the pipeline manager
    constructs the detector and the embedder, so the models load and warm up here
    rather than on the first tracked frame. Subsequent cameras reuse them.
    """
    container.pipeline_manager.start(camera_id)
    _log.info("api.tracking_started", camera_id=camera_id)
    return TrackingStatus.model_validate(container.pipeline_manager.status(camera_id))


@router.post(
    "/{camera_id}/stop",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Stop tracking on a camera",
    response_class=Response,
)
def stop_tracking(container: ContainerDep, camera_id: CameraId) -> Response:
    """Stop a camera's pipeline and release its per-track identity state.

    The camera itself stays open, so it can be inspected or restarted without
    reconnecting to the source.
    """
    container.pipeline_manager.stop(camera_id)
    _log.info("api.tracking_stopped", camera_id=camera_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{camera_id}/status", response_model=TrackingStatus, summary="Get one pipeline status")
def get_tracking(container: ContainerDep, camera_id: CameraId) -> TrackingStatus:
    """Return one pipeline's state and rolling metrics."""
    return TrackingStatus.model_validate(container.pipeline_manager.status(camera_id))


@router.get(
    "/{camera_id}/stream",
    summary="Live annotated video as MJPEG",
    response_class=StreamingResponse,
    responses={
        status.HTTP_200_OK: {
            "content": {"multipart/x-mixed-replace": {}},
            "description": "Annotated frames, one JPEG per part.",
        }
    },
)
def stream(container: ContainerDep, camera_id: CameraId) -> StreamingResponse:
    """Stream a camera's annotated frames to the browser.

    Resolve the source before returning the response, so a request for a camera
    that is not being tracked fails with a clean 404 instead of opening a stream
    that would never produce a frame.
    """
    source = container.pipeline_manager.frame_source(camera_id)
    return StreamingResponse(
        mjpeg_stream(source, fps=container.settings.camera.target_fps, camera_id=camera_id),
        media_type=MJPEG_CONTENT_TYPE,
        headers={
            # Proxies and browsers both like to buffer; either would turn a live
            # stream into a stuttering one that arrives in chunks.
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "X-Accel-Buffering": "no",
            "Pragma": "no-cache",
        },
    )
