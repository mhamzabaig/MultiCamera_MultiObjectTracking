"""Request and response schemas for the persons endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "GalleryStats",
    "PersonResponse",
    "PersonsResponse",
    "RebuildResponse",
    "RenameRequest",
]


class PersonResponse(BaseModel):
    """One enrolled identity."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    created_at: str
    updated_at: str
    embedding_count: int = Field(description="How many gallery vectors this person has.")
    representative_image: str | None = Field(
        default=None,
        description=(
            "Thumbnail path relative to the blob directory, or null if none was kept. "
            "Fetch the image itself from /api/persons/{id}/image."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class GalleryStats(BaseModel):
    """Counts across the gallery and the live vote state."""

    model_config = ConfigDict(extra="allow")

    persons: int
    embeddings: int
    index_vectors: int = Field(
        description=(
            "Vectors in the similarity index. Should equal `embeddings`; a mismatch "
            "means the derived index has diverged and should be rebuilt."
        )
    )


class PersonsResponse(BaseModel):
    """Every enrolled person, with gallery totals."""

    model_config = ConfigDict(extra="forbid")

    persons: list[PersonResponse]
    stats: GalleryStats


class RenameRequest(BaseModel):
    """A new display name."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120, description="New display name; must be unique.")


class RebuildResponse(BaseModel):
    """Result of rebuilding the similarity index from the database."""

    model_config = ConfigDict(extra="forbid")

    vectors: int = Field(description="Vectors indexed after the rebuild.")
