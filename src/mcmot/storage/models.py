"""Relational schema: two tables.

``persons`` holds enrolled identities. ``embeddings`` holds their gallery vectors,
stored as raw ``float32`` bytes in a BLOB column.

Why the vectors live in the database rather than in a separate vector file: the
database becomes the single source of truth, so the FAISS index is a derived cache
that can be rebuilt at any time. 512 floats is 2 KB, and 20 per person is 40 KB —
small enough that a separate file store would add a consistency problem for no
saving.

``embeddings.vector_id`` is an autoincrement integer that doubles as the FAISS
identifier. One key, two stores, and deleting a person removes it from both.

Kept portable for a future PostgreSQL move: no SQLite-specific types, explicit
lengths on string columns, and timestamps stored as naive UTC (see
:mod:`mcmot.storage.repository` for the conversion, since SQLite cannot preserve
a timezone offset).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ["Base", "EmbeddingRow", "PersonRow"]


def _utc_naive_now() -> datetime:
    """Return the current UTC time without a tzinfo, as stored."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Declarative base for every table."""


class PersonRow(Base):
    """An enrolled identity."""

    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True, index=True)
    """Unique on purpose. Two rows named "Hamza" would compete for every match and
    depress the decision margin, so a repeat registration is rejected as an error
    the UI can explain rather than silently degrading recognition."""

    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utc_naive_now, onupdate=_utc_naive_now
    )
    representative_image: Mapped[str | None] = mapped_column(String(400), nullable=True)
    """Path relative to ``storage.blob_dir``, so moving the data directory does not
    invalidate every row."""

    embeddings: Mapped[list[EmbeddingRow]] = relationship(
        back_populates="person",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class EmbeddingRow(Base):
    """One gallery embedding belonging to a person."""

    __tablename__ = "embeddings"

    vector_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    """Also the FAISS identifier. Never reused, so a stale index entry can always
    be recognised rather than silently pointing at a different person."""

    person_id: Mapped[str] = mapped_column(
        ForeignKey("persons.id", ondelete="CASCADE"), nullable=False, index=True
    )
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    """Raw little-endian ``float32`` bytes, L2-normalised by the embedder."""

    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    """Stored explicitly so a model swap that changes dimensionality is detectable
    instead of producing a garbled reinterpretation of the bytes."""

    quality: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utc_naive_now)

    person: Mapped[PersonRow] = relationship(back_populates="embeddings")
