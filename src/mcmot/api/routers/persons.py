"""Person endpoints: registration, listing, renaming, deletion, index maintenance.

**Which handlers are async.** Only the ones that actually await something — reading
an upload off the wire. Everything else is a plain ``def``, which FastAPI runs in its
own threadpool, so a SQLite query or a GPU forward pass cannot block the event loop
and stall every MJPEG stream in the process. Registration does both: it awaits the
upload, then hands the CPU-and-GPU work to a worker thread explicitly.

No error mapping lives here. Every failure the services raise is an ``MCMOTError``
carrying its own status — unknown person 404, duplicate name 409, unusable frames
422 — and the handler registered in ``api/errors.py`` turns that into the response.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Path, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from mcmot.api.deps import ContainerDep
from mcmot.api.schemas.persons import (
    GalleryStats,
    PersonResponse,
    PersonsResponse,
    RebuildResponse,
    RenameRequest,
)
from mcmot.api.schemas.system import ErrorResponse
from mcmot.core.errors import PersonNotFoundError
from mcmot.core.logging import get_logger

__all__ = ["router"]

_log = get_logger(__name__)

PersonId = Annotated[str, Path(description="Person identifier.")]

router = APIRouter(
    prefix="/api/persons",
    tags=["persons"],
    responses={
        status.HTTP_404_NOT_FOUND: {"model": ErrorResponse, "description": "No such person."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)


@router.get("", response_model=PersonsResponse, summary="List enrolled people")
def list_persons(container: ContainerDep) -> PersonsResponse:
    """Return every enrolled person and the gallery totals.

    The totals include both the database count and the index count so the UI can
    show divergence rather than hide it: they should be equal, and an inequality is
    the signal to rebuild.
    """
    service = container.identity_service
    return PersonsResponse(
        persons=[
            PersonResponse.model_validate(person.to_dict()) for person in service.list_persons()
        ],
        stats=GalleryStats.model_validate(service.stats()),
    )


@router.post(
    "",
    response_model=PersonResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new person from captured frames",
    responses={
        status.HTTP_409_CONFLICT: {"model": ErrorResponse, "description": "Name already taken."},
        status.HTTP_422_UNPROCESSABLE_ENTITY: {
            "model": ErrorResponse,
            "description": "Too few usable frames, or no person detected in any of them.",
        },
    },
)
async def register_person(
    container: ContainerDep,
    name: Annotated[str, Form(min_length=1, max_length=120, description="Display name.")],
    frames: Annotated[list[UploadFile], File(description="Captured frames, JPEG or PNG.")],
) -> PersonResponse:
    """Enrol a person from whole captured frames.

    Send the frames as they came off the camera; the server detects the person,
    crops them, filters for quality, and keeps a diverse subset. Twenty to fifty
    frames covering a few angles is the intended input.
    """
    payloads = [await frame.read() for frame in frames]
    _log.info("api.register_requested", name=name, frames=len(payloads))

    # Decode, detect, embed: hundreds of milliseconds of blocking CPU and GPU work.
    person = await run_in_threadpool(container.registration_service.register, name, payloads)
    return PersonResponse.model_validate(person.to_dict())


@router.get("/{person_id}", response_model=PersonResponse, summary="Get one person")
def get_person(container: ContainerDep, person_id: PersonId) -> PersonResponse:
    """Return one enrolled person."""
    return PersonResponse.model_validate(container.identity_service.get_person(person_id).to_dict())


@router.get(
    "/{person_id}/image",
    summary="Get a person's representative image",
    response_class=Response,
    responses={
        status.HTTP_200_OK: {"content": {"image/jpeg": {}}, "description": "The thumbnail."},
        status.HTTP_404_NOT_FOUND: {
            "model": ErrorResponse,
            "description": "No such person, or no image was kept for them.",
        },
    },
)
def get_person_image(container: ContainerDep, person_id: PersonId) -> Response:
    """Serve the stored thumbnail for a person.

    Served through the API rather than as a static mount: the blob directory holds
    images keyed by person id, and exposing it wholesale would make every
    registration photo readable by anyone who can guess a path.
    """
    person = container.identity_service.get_person(person_id)
    if not person.representative_image:
        raise PersonNotFoundError(
            "No representative image was kept for this person.",
            details={"person_id": person_id},
        )
    payload = container.blob_store.read(person.representative_image)
    # Immutable once written, so it is safe to cache hard; the filename changes when
    # the image does.
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.patch("/{person_id}", response_model=PersonResponse, summary="Rename a person")
def rename_person(
    container: ContainerDep, person_id: PersonId, payload: RenameRequest
) -> PersonResponse:
    """Change a person's display name.

    Only the name: embeddings are unaffected, because identity is the vector
    gallery and the name is just its label.
    """
    person = container.identity_service.rename_person(person_id, payload.name)
    return PersonResponse.model_validate(person.to_dict())


@router.delete(
    "/{person_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a person",
    response_class=Response,
)
def delete_person(container: ContainerDep, person_id: PersonId) -> Response:
    """Remove a person from the database, the index, the blob store, and live votes.

    Deleting takes effect immediately on running cameras: a track that was showing
    this name stops showing it on its next verdict, rather than displaying someone
    who is no longer enrolled.
    """
    container.identity_service.delete_person(person_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/index/rebuild",
    response_model=RebuildResponse,
    summary="Rebuild the similarity index from the database",
)
def rebuild_index(container: ContainerDep) -> RebuildResponse:
    """Discard and rebuild the vector index from the database.

    The recovery path whenever the two stores disagree. Safe to call at any time:
    the database is the source of truth and the index is a derived cache, so the
    worst case is a few seconds of rebuild.
    """
    return RebuildResponse(vectors=container.identity_service.rebuild_index())
