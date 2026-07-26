"""Enrolled person and gallery embedding types.

These are the *domain* representations, deliberately independent of SQLAlchemy.
Phase 2 introduces ORM entities that map onto these, so services, the API layer,
and their tests depend on plain dataclasses rather than on live database rows —
which is also what keeps the PostgreSQL migration a storage-layer concern.

Identifiers are opaque strings (UUID4 hex). Not autoincrement integers: they are
generated before the row is written, they do not collide when two cameras or two
future service instances enrol concurrently, and they carry no accidental
ordering meaning.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from mcmot.domain.types import EmbeddingArray

__all__ = ["EmbeddingRecord", "PersonRecord", "new_person_id", "utc_now"]


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Always aware, never naive: SQLite happily stores naive datetimes and then
    nothing can tell whether a timestamp was local or UTC.

    ``timezone.utc`` rather than ``datetime.UTC`` — the latter is 3.11+, and this
    project targets 3.10.
    """
    return datetime.now(timezone.utc)


def new_person_id() -> str:
    """Generate an opaque person identifier."""
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True, eq=False)
class EmbeddingRecord:
    """One gallery embedding belonging to an enrolled person."""

    vector_id: int
    """Stable ``int64`` key shared with the vector index. The index maps this to a
    vector; the database maps it to a person. Deleting a person removes its
    ``vector_id`` values from both, which is what keeps the two stores aligned."""

    person_id: str
    vector: EmbeddingArray
    """L2-normalised ``float32`` embedding of shape ``(embedding_dim,)``.
    Normalisation is the embedder's responsibility and is what makes inner
    product identical to cosine similarity in the index."""

    quality: float = 1.0
    """Quality score of the source crop, in ``[0, 1]``. Retained so that when a
    person exceeds ``identity.max_embeddings_per_person`` the weakest embedding
    is evicted rather than the oldest."""

    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the vector's shape and dtype at the boundary."""
        if self.vector.ndim != 1:
            raise ValueError(f"Embedding must be 1-D, got shape {self.vector.shape!r}")
        if self.vector.dtype != np.float32:
            raise ValueError(f"Embedding must be float32, got {self.vector.dtype}")

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.vector.shape[0])


@dataclass(frozen=True, slots=True)
class PersonRecord:
    """An enrolled identity."""

    id: str = field(default_factory=new_person_id)
    name: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    embedding_count: int = 0
    representative_image: str | None = None
    """Path of the display thumbnail, relative to ``storage.blob_dir``.

    A relative path, so moving or backing up the data directory does not
    invalidate every row — and so images stay on disk rather than as BLOBs,
    which keeps the database small and the PostgreSQL migration trivial."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form attributes (department, notes, external ids). Present so that
    adding a field does not require a schema migration for every deployment."""

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "embedding_count": self.embedding_count,
            "representative_image": self.representative_image,
            "metadata": dict(self.metadata),
        }
