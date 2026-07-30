"""Camera endpoints: open, list, inspect, close.

Opening a camera and *tracking* on it are separate operations, and deliberately so.
A camera can be connected and healthy while nothing is being tracked on it, which is
what the Cameras page shows; starting the pipeline is a second, more expensive step
that the Tracking page owns. Conflating them would mean you could not check whether
a phone's stream URL is even reachable without also loading the models.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Path, Response, status

from mcmot.api.deps import ContainerDep
from mcmot.api.schemas.cameras import CameraResponse, CamerasResponse, OpenCameraRequest
from mcmot.api.schemas.system import ErrorResponse
from mcmot.core.logging import get_logger

__all__ = ["router"]

_log = get_logger(__name__)

CameraId = Annotated[str, Path(description="Camera identifier.")]

router = APIRouter(
    prefix="/api/cameras",
    tags=["cameras"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "Camera is not open."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.get("", response_model=CamerasResponse, summary="List open cameras")
def list_cameras(container: ContainerDep) -> CamerasResponse:
    """Return every open camera with its live capture statistics."""
    return CamerasResponse(
        cameras=[
            CameraResponse.model_validate(status_dict)
            for status_dict in container.camera_manager.statuses()
        ]
    )


@router.post(
    "",
    response_model=CameraResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Open a camera",
    responses={
        status.HTTP_409_CONFLICT: {
            "model": ErrorResponse,
            "description": "A camera is already open under this id.",
        },
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorResponse,
            "description": "The source string is not a recognisable camera.",
        },
    },
)
def open_camera(container: ContainerDep, payload: OpenCameraRequest) -> CameraResponse:
    """Open a camera and start capturing in the background.

    Returns as soon as the capture thread starts, **before** the stream is known to
    be working. That is intentional: a phone that has not launched its camera app yet
    should not fail the request, it should show as ``connecting`` and then reconnect
    on its own. Poll this camera's status to see whether it reached ``streaming``.
    """
    container.camera_manager.open(payload.camera_id, payload.source)
    _log.info("api.camera_opened", camera_id=payload.camera_id)
    return CameraResponse.model_validate(container.camera_manager.status(payload.camera_id))


@router.get("/{camera_id}", response_model=CameraResponse, summary="Get one camera's status")
def get_camera(container: ContainerDep, camera_id: CameraId) -> CameraResponse:
    """Return one camera's specification and capture statistics."""
    return CameraResponse.model_validate(container.camera_manager.status(camera_id))


@router.delete(
    "/{camera_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Close a camera",
    response_class=Response,
)
def close_camera(container: ContainerDep, camera_id: CameraId) -> Response:
    """Stop and remove a camera, stopping its pipeline first if one is running.

    Closing the camera out from under a running pipeline would leave a worker
    spinning against a source that will never produce another frame, so the
    pipeline is stopped first — the same ordering the container uses at shutdown.
    """
    if container.pipeline_manager.is_running(camera_id):
        container.pipeline_manager.stop(camera_id)
    container.camera_manager.close(camera_id)
    _log.info("api.camera_closed", camera_id=camera_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
