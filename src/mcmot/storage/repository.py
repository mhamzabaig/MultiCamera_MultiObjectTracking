"""Person and embedding persistence.

The only module that knows SQLAlchemy exists. Everything above it receives domain
dataclasses (:class:`~mcmot.domain.PersonRecord`,
:class:`~mcmot.domain.EmbeddingRecord`), never live ORM rows, which keeps
detached-instance surprises out of the pipeline and makes the storage backend
genuinely replaceable.

Deletion returns the affected ``vector_id`` values rather than touching the vector
index itself. The repository owns one store; the caller coordinates both. That
keeps the ordering decision — remove from the index only after the database commit
succeeded — visible at the call site instead of buried here.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from http import HTTPStatus

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from mcmot.core.errors import PersonNotFoundError, StorageError
from mcmot.core.logging import get_logger
from mcmot.domain.person import EmbeddingRecord, PersonRecord, new_person_id
from mcmot.domain.types import EmbeddingArray, EmbeddingMatrix
from mcmot.storage.db import Database
from mcmot.storage.models import EmbeddingRow, PersonRow

__all__ = ["PersonAlreadyExistsError", "PersonRepository"]

_log = get_logger(__name__)


class PersonAlreadyExistsError(StorageError):
    """A person with this name is already enrolled."""

    code = "person.already_exists"
    http_status = HTTPStatus.CONFLICT
    default_message = "A person with this name is already registered."


def _as_utc(value: datetime) -> datetime:
    """Re-attach UTC to a timestamp read back from SQLite.

    Values are written as naive UTC because SQLite cannot store an offset. Marking
    them aware on the way out means nothing downstream ever has to guess whether a
    timestamp was local time.
    """
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class PersonRepository:
    """Reads and writes enrolled people and their gallery embeddings."""

    def __init__(self, database: Database) -> None:
        """Args: database: Configured database wrapper."""
        self._db = database

    # -- people -----------------------------------------------------------
    def create(self, name: str, *, representative_image: str | None = None) -> PersonRecord:
        """Enrol a new person.

        Args:
            name: Display name; must be unique and non-blank.
            representative_image: Thumbnail path relative to the blob directory.

        Returns:
            The created record.

        Raises:
            PersonAlreadyExistsError: If the name is taken.
            StorageError: If the name is blank or the write fails.
        """
        clean = name.strip()
        if not clean:
            raise StorageError("Person name must not be empty.", details={"name": name})

        row = PersonRow(id=new_person_id(), name=clean, representative_image=representative_image)
        try:
            with self._db.session() as session:
                session.add(row)
                session.flush()
                record = self._to_person(row, embedding_count=0)
        except StorageError as exc:
            # The unique constraint is the expected failure here; surface it as a
            # distinct code so the UI can say "already registered".
            if isinstance(exc.__cause__, IntegrityError):
                raise PersonAlreadyExistsError(details={"name": clean}) from exc
            raise

        _log.info("person.created", person_id=record.id, name=record.name)
        return record

    def get(self, person_id: str) -> PersonRecord:
        """Return one person.

        Raises:
            PersonNotFoundError: If no such person exists.
        """
        with self._db.session() as session:
            row = session.get(PersonRow, person_id)
            if row is None:
                raise PersonNotFoundError(details={"person_id": person_id})
            return self._to_person(row, embedding_count=self._count_embeddings(session, person_id))

    def get_by_name(self, name: str) -> PersonRecord | None:
        """Return the person with this name, or ``None``."""
        with self._db.session() as session:
            row = session.scalar(select(PersonRow).where(PersonRow.name == name.strip()))
            if row is None:
                return None
            return self._to_person(row, embedding_count=self._count_embeddings(session, row.id))

    def list_all(self) -> list[PersonRecord]:
        """Return every person, newest first, each with its embedding count."""
        with self._db.session() as session:
            # One grouped count instead of a query per person: the Database page
            # lists everyone, and N+1 queries there would be felt immediately.
            counts: dict[str, int] = {
                str(person_id): int(total)
                for person_id, total in session.execute(
                    select(EmbeddingRow.person_id, func.count(EmbeddingRow.vector_id)).group_by(
                        EmbeddingRow.person_id
                    )
                ).all()
            }
            rows = session.scalars(
                select(PersonRow).order_by(PersonRow.created_at.desc(), PersonRow.name)
            ).all()
            return [self._to_person(row, embedding_count=counts.get(row.id, 0)) for row in rows]

    def count(self) -> int:
        """Return the number of enrolled people."""
        with self._db.session() as session:
            return int(session.scalar(select(func.count(PersonRow.id))) or 0)

    def rename(self, person_id: str, name: str) -> PersonRecord:
        """Change a person's display name.

        Raises:
            PersonNotFoundError: If no such person exists.
            PersonAlreadyExistsError: If the new name is taken.
            StorageError: If the name is blank.
        """
        clean = name.strip()
        if not clean:
            raise StorageError("Person name must not be empty.", details={"name": name})

        try:
            with self._db.session() as session:
                row = session.get(PersonRow, person_id)
                if row is None:
                    raise PersonNotFoundError(details={"person_id": person_id})
                row.name = clean
                session.flush()
                record = self._to_person(
                    row, embedding_count=self._count_embeddings(session, person_id)
                )
        except StorageError as exc:
            if isinstance(exc.__cause__, IntegrityError):
                raise PersonAlreadyExistsError(details={"name": clean}) from exc
            raise

        _log.info("person.renamed", person_id=person_id, name=clean)
        return record

    def set_representative_image(self, person_id: str, relative_path: str | None) -> None:
        """Point a person at their display thumbnail.

        Raises:
            PersonNotFoundError: If no such person exists.
        """
        with self._db.session() as session:
            row = session.get(PersonRow, person_id)
            if row is None:
                raise PersonNotFoundError(details={"person_id": person_id})
            row.representative_image = relative_path

    def delete(self, person_id: str) -> list[int]:
        """Delete a person and their embeddings.

        Returns:
            The ``vector_id`` values that were removed, so the caller can drop the
            same ids from the vector index once this commit has succeeded.

        Raises:
            PersonNotFoundError: If no such person exists.
        """
        with self._db.session() as session:
            row = session.get(PersonRow, person_id)
            if row is None:
                raise PersonNotFoundError(details={"person_id": person_id})
            vector_ids = list(
                session.scalars(
                    select(EmbeddingRow.vector_id).where(EmbeddingRow.person_id == person_id)
                ).all()
            )
            session.delete(row)

        _log.info("person.deleted", person_id=person_id, embeddings=len(vector_ids))
        return vector_ids

    # -- embeddings -------------------------------------------------------
    def add_embeddings(
        self,
        person_id: str,
        vectors: EmbeddingMatrix,
        qualities: Sequence[float] | None = None,
    ) -> list[EmbeddingRecord]:
        """Append gallery embeddings for a person.

        Args:
            person_id: Owner of the embeddings.
            vectors: ``float32`` array of shape ``(n, dim)``, L2-normalised.
            qualities: Optional per-vector quality in ``[0, 1]``; defaults to 1.0.
                Retained so eviction can drop the weakest embedding rather than
                the oldest.

        Returns:
            The created records, carrying their assigned ``vector_id`` values for
            insertion into the vector index.

        Raises:
            PersonNotFoundError: If no such person exists.
            StorageError: If the array shape, dtype, or quality count is wrong.
        """
        matrix = self._validate_matrix(vectors)
        count = matrix.shape[0]
        scores = self._validate_qualities(qualities, count)

        with self._db.session() as session:
            if session.get(PersonRow, person_id) is None:
                raise PersonNotFoundError(details={"person_id": person_id})

            dim = int(matrix.shape[1])
            rows = [
                EmbeddingRow(
                    person_id=person_id,
                    vector=matrix[i].tobytes(),
                    dim=dim,
                    quality=float(scores[i]),
                )
                for i in range(count)
            ]
            session.add_all(rows)
            # Flush to obtain the autoincrement ids before the session closes.
            session.flush()
            records = [
                EmbeddingRecord(
                    vector_id=row.vector_id,
                    person_id=person_id,
                    vector=matrix[i],
                    quality=row.quality,
                )
                for i, row in enumerate(rows)
            ]

        _log.info("embeddings.added", person_id=person_id, count=count, dim=dim)
        return records

    def list_embeddings(self, person_id: str | None = None) -> list[EmbeddingRecord]:
        """Return embeddings for one person, or for everyone when ``person_id`` is None.

        The full listing is what rebuilds the vector index from the source of truth.
        """
        with self._db.session() as session:
            statement = select(EmbeddingRow).order_by(EmbeddingRow.vector_id)
            if person_id is not None:
                statement = statement.where(EmbeddingRow.person_id == person_id)
            return [self._to_embedding(row) for row in session.scalars(statement).all()]

    def count_embeddings(self) -> int:
        """Return the total number of stored embeddings.

        Compared against the vector index's size at startup to detect divergence.
        """
        with self._db.session() as session:
            return int(session.scalar(select(func.count(EmbeddingRow.vector_id))) or 0)

    def delete_embeddings(self, vector_ids: Sequence[int]) -> int:
        """Delete specific embeddings, e.g. when evicting the weakest of a person's set.

        Returns:
            The number of rows actually deleted.
        """
        if not vector_ids:
            return 0
        with self._db.session() as session:
            rows = session.scalars(
                select(EmbeddingRow).where(EmbeddingRow.vector_id.in_(list(vector_ids)))
            ).all()
            for row in rows:
                session.delete(row)
            return len(rows)

    def owners_of(self, vector_ids: Sequence[int]) -> dict[int, tuple[str, str]]:
        """Map ``vector_id`` to ``(person_id, name)``.

        This is the lookup that turns a vector-index search result into a name.
        Returns only ids that exist, so an index entry left behind by a failed
        delete resolves to nothing rather than to the wrong person.
        """
        if not vector_ids:
            return {}
        with self._db.session() as session:
            rows = session.execute(
                select(EmbeddingRow.vector_id, PersonRow.id, PersonRow.name)
                .join(PersonRow, PersonRow.id == EmbeddingRow.person_id)
                .where(EmbeddingRow.vector_id.in_(list(vector_ids)))
            ).all()
            return {int(vid): (pid, name) for vid, pid, name in rows}

    # -- validation and mapping -------------------------------------------
    @staticmethod
    def _validate_matrix(vectors: EmbeddingMatrix) -> EmbeddingMatrix:
        if vectors.ndim != 2:
            raise StorageError(
                "Embeddings must be a 2-D (n, dim) array.",
                details={"shape": list(vectors.shape)},
            )
        if vectors.shape[0] == 0:
            raise StorageError("No embeddings supplied.", details={"shape": list(vectors.shape)})
        if vectors.dtype != np.float32:
            raise StorageError("Embeddings must be float32.", details={"dtype": str(vectors.dtype)})
        # Contiguity matters: tobytes() on a sliced view would silently store the
        # wrong bytes.
        return np.ascontiguousarray(vectors)

    @staticmethod
    def _validate_qualities(qualities: Sequence[float] | None, count: int) -> Sequence[float]:
        if qualities is None:
            return [1.0] * count
        if len(qualities) != count:
            raise StorageError(
                "One quality score is required per embedding.",
                details={"embeddings": count, "qualities": len(qualities)},
            )
        return qualities

    @staticmethod
    def _count_embeddings(session: Session, person_id: str) -> int:
        return int(
            session.scalar(
                select(func.count(EmbeddingRow.vector_id)).where(
                    EmbeddingRow.person_id == person_id
                )
            )
            or 0
        )

    @staticmethod
    def _to_person(row: PersonRow, *, embedding_count: int) -> PersonRecord:
        return PersonRecord(
            id=row.id,
            name=row.name,
            created_at=_as_utc(row.created_at),
            updated_at=_as_utc(row.updated_at),
            embedding_count=embedding_count,
            representative_image=row.representative_image,
        )

    @staticmethod
    def _to_embedding(row: EmbeddingRow) -> EmbeddingRecord:
        vector: EmbeddingArray = np.frombuffer(row.vector, dtype=np.float32)
        if vector.shape[0] != row.dim:
            raise StorageError(
                "Stored embedding length does not match its recorded dimension.",
                details={"vector_id": row.vector_id, "expected": row.dim, "found": len(vector)},
            )
        return EmbeddingRecord(
            vector_id=row.vector_id,
            person_id=row.person_id,
            # frombuffer returns a read-only view of the BLOB; copy so callers
            # (and FAISS) receive a writable, independently-owned array.
            vector=vector.copy(),
            quality=row.quality,
            created_at=_as_utc(row.created_at),
        )
