"""Storage tests: database, repository, vector index, blob store, consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from mcmot.core.errors import PersonNotFoundError, StorageError, VectorIndexError
from mcmot.core.settings import StorageSettings
from mcmot.storage.blobs import BlobStore
from mcmot.storage.db import Database
from mcmot.storage.models import PersonRow
from mcmot.storage.repository import PersonAlreadyExistsError, PersonRepository
from mcmot.storage.vector_index import FaissIndex

DIM = 8


def unit_vectors(count: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Return `count` random L2-normalised float32 vectors."""
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, dim)).astype(np.float32)
    return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)


@pytest.fixture
def database(tmp_path: Path) -> Database:
    db = Database(StorageSettings(database_url=f"sqlite:///{(tmp_path / 'test.db').as_posix()}"))
    db.create_schema()
    yield db
    db.dispose()


@pytest.fixture
def repo(database: Database) -> PersonRepository:
    return PersonRepository(database)


@pytest.fixture
def index(tmp_path: Path) -> FaissIndex:
    return FaissIndex(tmp_path / "index" / "gallery.faiss", dim=DIM)


@pytest.fixture
def blobs(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "persons")


class TestDatabase:
    def test_schema_creation_is_idempotent(self, database: Database) -> None:
        database.create_schema()

        with database.session() as session:
            tables = set(
                session.scalars(text("SELECT name FROM sqlite_master WHERE type='table'")).all()
            )
        assert {"persons", "embeddings"} <= tables

    def test_pragmas_are_applied(self, database: Database) -> None:
        """foreign_keys=ON is not optional: without it ON DELETE CASCADE is inert."""
        with database.session() as session:
            assert session.scalar(text("PRAGMA foreign_keys")) == 1
            assert str(session.scalar(text("PRAGMA journal_mode"))).lower() == "wal"

    def test_parent_directory_is_created(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper" / "mcmot.db"
        db = Database(StorageSettings(database_url=f"sqlite:///{target.as_posix()}"))
        db.create_schema()

        assert target.is_file()
        db.dispose()

    def test_session_rolls_back_on_error(self, database: Database, repo: PersonRepository) -> None:
        """A write followed by an exception must leave nothing behind."""
        with pytest.raises(RuntimeError), database.session() as session:
            session.add(PersonRow(id="x" * 32, name="Ghost"))
            session.flush()
            raise RuntimeError("boom")

        assert repo.count() == 0


class TestPersonCrud:
    def test_create_and_get(self, repo: PersonRepository) -> None:
        created = repo.create("Hamza")

        fetched = repo.get(created.id)

        assert fetched.name == "Hamza"
        assert fetched.embedding_count == 0
        assert fetched.created_at.tzinfo is not None

    def test_name_is_trimmed(self, repo: PersonRepository) -> None:
        assert repo.create("  Hamza  ").name == "Hamza"

    def test_blank_name_is_rejected(self, repo: PersonRepository) -> None:
        with pytest.raises(StorageError, match="must not be empty"):
            repo.create("   ")

    def test_duplicate_name_is_rejected(self, repo: PersonRepository) -> None:
        """Two rows with one name would compete for every match and depress margins."""
        repo.create("Hamza")

        with pytest.raises(PersonAlreadyExistsError) as excinfo:
            repo.create("Hamza")

        assert excinfo.value.code == "person.already_exists"

    def test_get_missing_person_raises(self, repo: PersonRepository) -> None:
        with pytest.raises(PersonNotFoundError):
            repo.get("does-not-exist")

    def test_get_by_name(self, repo: PersonRepository) -> None:
        repo.create("Hamza")

        assert repo.get_by_name("Hamza") is not None
        assert repo.get_by_name("Nobody") is None

    def test_list_all_reports_embedding_counts(self, repo: PersonRepository) -> None:
        first = repo.create("Hamza")
        repo.create("Ali")
        repo.add_embeddings(first.id, unit_vectors(3))

        by_name = {person.name: person for person in repo.list_all()}

        assert by_name["Hamza"].embedding_count == 3
        assert by_name["Ali"].embedding_count == 0

    def test_rename(self, repo: PersonRepository) -> None:
        person = repo.create("Hamza")

        assert repo.rename(person.id, "Hamza Ali").name == "Hamza Ali"

    def test_rename_to_a_taken_name_is_rejected(self, repo: PersonRepository) -> None:
        repo.create("Hamza")
        other = repo.create("Ali")

        with pytest.raises(PersonAlreadyExistsError):
            repo.rename(other.id, "Hamza")

    def test_delete_returns_vector_ids_and_cascades(self, repo: PersonRepository) -> None:
        """The caller needs the ids to drop the same vectors from the index."""
        person = repo.create("Hamza")
        records = repo.add_embeddings(person.id, unit_vectors(4))

        removed = repo.delete(person.id)

        assert sorted(removed) == sorted(r.vector_id for r in records)
        assert repo.count_embeddings() == 0
        with pytest.raises(PersonNotFoundError):
            repo.get(person.id)

    def test_delete_missing_person_raises(self, repo: PersonRepository) -> None:
        with pytest.raises(PersonNotFoundError):
            repo.delete("nope")


class TestEmbeddings:
    def test_add_returns_records_with_ids(self, repo: PersonRepository) -> None:
        person = repo.create("Hamza")

        records = repo.add_embeddings(person.id, unit_vectors(3))

        assert len(records) == 3
        assert all(r.vector_id > 0 for r in records)
        assert len({r.vector_id for r in records}) == 3

    def test_vectors_survive_the_blob_round_trip(self, repo: PersonRepository) -> None:
        """A byte-order or contiguity mistake here would silently corrupt matching."""
        person = repo.create("Hamza")
        vectors = unit_vectors(2)

        repo.add_embeddings(person.id, vectors)
        stored = repo.list_embeddings(person.id)

        np.testing.assert_allclose(stored[0].vector, vectors[0], rtol=0, atol=0)
        np.testing.assert_allclose(stored[1].vector, vectors[1], rtol=0, atol=0)
        assert stored[0].vector.dtype == np.float32
        assert stored[0].dim == DIM

    def test_non_contiguous_input_is_stored_correctly(self, repo: PersonRepository) -> None:
        person = repo.create("Hamza")
        view = unit_vectors(4)[::2]  # a strided view, not contiguous
        assert not view.flags["C_CONTIGUOUS"]

        repo.add_embeddings(person.id, view)

        np.testing.assert_allclose(repo.list_embeddings(person.id)[0].vector, view[0])

    def test_qualities_are_stored(self, repo: PersonRepository) -> None:
        person = repo.create("Hamza")

        repo.add_embeddings(person.id, unit_vectors(2), [0.4, 0.9])

        assert [r.quality for r in repo.list_embeddings(person.id)] == [0.4, 0.9]

    def test_quality_count_must_match(self, repo: PersonRepository) -> None:
        person = repo.create("Hamza")

        with pytest.raises(StorageError, match="One quality score"):
            repo.add_embeddings(person.id, unit_vectors(3), [1.0])

    @pytest.mark.parametrize(
        ("vectors", "match"),
        [
            (np.zeros((0, DIM), dtype=np.float32), "No embeddings"),
            (np.zeros(DIM, dtype=np.float32), "2-D"),
            (np.zeros((2, DIM), dtype=np.float64), "float32"),
        ],
        ids=["empty", "one-dimensional", "float64"],
    )
    def test_invalid_arrays_are_rejected(
        self, repo: PersonRepository, vectors: np.ndarray, match: str
    ) -> None:
        person = repo.create("Hamza")

        with pytest.raises(StorageError, match=match):
            repo.add_embeddings(person.id, vectors)

    def test_add_for_unknown_person_raises(self, repo: PersonRepository) -> None:
        with pytest.raises(PersonNotFoundError):
            repo.add_embeddings("nope", unit_vectors(1))

    def test_owners_of_maps_vectors_to_people(self, repo: PersonRepository) -> None:
        """This lookup is what turns an index hit into a name."""
        hamza = repo.create("Hamza")
        ali = repo.create("Ali")
        hamza_vectors = repo.add_embeddings(hamza.id, unit_vectors(2, seed=1))
        ali_vectors = repo.add_embeddings(ali.id, unit_vectors(1, seed=2))

        owners = repo.owners_of([hamza_vectors[0].vector_id, ali_vectors[0].vector_id, 9999])

        assert owners[hamza_vectors[0].vector_id] == (hamza.id, "Hamza")
        assert owners[ali_vectors[0].vector_id] == (ali.id, "Ali")
        assert 9999 not in owners  # a stale index entry resolves to nobody

    def test_owners_of_empty_input(self, repo: PersonRepository) -> None:
        assert repo.owners_of([]) == {}

    def test_delete_specific_embeddings(self, repo: PersonRepository) -> None:
        person = repo.create("Hamza")
        records = repo.add_embeddings(person.id, unit_vectors(3))

        deleted = repo.delete_embeddings([records[0].vector_id, records[2].vector_id])

        assert deleted == 2
        assert repo.count_embeddings() == 1


class TestVectorIndex:
    def test_add_and_search(self, index: FaissIndex) -> None:
        vectors = unit_vectors(3, seed=5)
        index.add([10, 20, 30], vectors)

        scores, ids = index.search(vectors[1:2], top_k=1)

        assert ids[0][0] == 20
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)
        assert index.size == 3

    def test_search_is_exact_cosine(self, index: FaissIndex) -> None:
        """Inner product on unit vectors must equal the cosine we expect."""
        a = np.array([[1.0, 0.0] + [0.0] * (DIM - 2)], dtype=np.float32)
        b = np.array([[0.0, 1.0] + [0.0] * (DIM - 2)], dtype=np.float32)
        index.add([1, 2], np.vstack([a, b]))

        scores, ids = index.search(a, top_k=2)

        ranked = dict(zip(ids[0].tolist(), scores[0].tolist(), strict=True))
        assert ranked[1] == pytest.approx(1.0, abs=1e-5)
        assert ranked[2] == pytest.approx(0.0, abs=1e-5)

    def test_search_on_an_empty_index_returns_sentinels(self, index: FaissIndex) -> None:
        scores, ids = index.search(unit_vectors(1), top_k=5)

        assert ids.shape == (1, 5)
        assert (ids == -1).all()
        assert (scores < 0).all()

    def test_fewer_vectors_than_top_k_pads_with_minus_one(self, index: FaissIndex) -> None:
        """Callers must filter on id >= 0; this pins that FAISS behaviour."""
        index.add([1], unit_vectors(1))

        _, ids = index.search(unit_vectors(1), top_k=4)

        assert ids[0][0] == 1
        assert (ids[0][1:] == -1).all()

    def test_remove(self, index: FaissIndex) -> None:
        index.add([1, 2, 3], unit_vectors(3))

        assert index.remove([2]) == 1
        assert index.size == 2

    def test_remove_is_idempotent(self, index: FaissIndex) -> None:
        """Two-store deletion may be retried; removing an absent id must not fail."""
        index.add([1], unit_vectors(1))
        index.remove([1])

        assert index.remove([1]) == 0
        assert index.remove([]) == 0

    def test_unnormalised_vectors_are_rejected(self, index: FaissIndex) -> None:
        """Without unit norm, inner product is not cosine and every score is a lie."""
        raw = np.full((1, DIM), 3.0, dtype=np.float32)

        with pytest.raises(VectorIndexError, match="L2-normalised"):
            index.add([1], raw)

    @pytest.mark.parametrize(
        ("vectors", "match"),
        [
            (np.zeros((1, DIM + 1), dtype=np.float32), "dimension does not match"),
            (np.zeros((0, DIM), dtype=np.float32), "No vectors"),
            (np.zeros(DIM, dtype=np.float32), "2-D"),
            (unit_vectors(1).astype(np.float64), "float32"),
        ],
        ids=["wrong-dim", "empty", "one-dimensional", "float64"],
    )
    def test_invalid_arrays_are_rejected(
        self, index: FaissIndex, vectors: np.ndarray, match: str
    ) -> None:
        with pytest.raises(VectorIndexError, match=match):
            index.add(list(range(len(vectors))), vectors)

    def test_id_count_must_match_vector_count(self, index: FaissIndex) -> None:
        with pytest.raises(VectorIndexError, match="One id is required"):
            index.add([1, 2], unit_vectors(3))

    def test_non_positive_top_k_is_rejected(self, index: FaissIndex) -> None:
        with pytest.raises(VectorIndexError, match="top_k must be positive"):
            index.search(unit_vectors(1), top_k=0)

    def test_save_and_load_round_trip(self, index: FaissIndex, tmp_path: Path) -> None:
        vectors = unit_vectors(4, seed=7)
        index.add([11, 22, 33, 44], vectors)
        index.save()

        reopened = FaissIndex(index.path, dim=DIM)
        assert reopened.load() is True

        assert reopened.size == 4
        _, ids = reopened.search(vectors[2:3], top_k=1)
        assert ids[0][0] == 33

    def test_load_reports_a_missing_file(self, index: FaissIndex) -> None:
        assert index.load() is False
        assert index.size == 0

    def test_save_leaves_no_temporary_file(self, index: FaissIndex) -> None:
        index.add([1], unit_vectors(1))
        index.save()

        assert not list(index.path.parent.glob("*.tmp"))

    def test_load_rejects_a_dimension_mismatch(self, index: FaissIndex) -> None:
        """Catches a changed ReID model instead of silently misinterpreting vectors."""
        index.add([1], unit_vectors(1))
        index.save()

        with pytest.raises(VectorIndexError, match="different embedding dimension"):
            FaissIndex(index.path, dim=DIM * 2).load()

    def test_load_rejects_a_corrupt_file(self, index: FaissIndex) -> None:
        index.path.parent.mkdir(parents=True, exist_ok=True)
        index.path.write_bytes(b"not a faiss index")

        with pytest.raises(VectorIndexError, match="Could not read"):
            index.load()

    def test_reset_empties_the_index(self, index: FaissIndex) -> None:
        index.add([1, 2], unit_vectors(2))

        index.reset()

        assert index.size == 0

    def test_zero_dimension_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(VectorIndexError, match="must be positive"):
            FaissIndex(tmp_path / "x.faiss", dim=0)


class TestRebuild:
    def test_rebuild_from_database(self, repo: PersonRepository, index: FaissIndex) -> None:
        """The recovery path: the index holds nothing the database cannot restore."""
        person = repo.create("Hamza")
        records = repo.add_embeddings(person.id, unit_vectors(3, seed=3))

        count = index.rebuild_from(repo.list_embeddings())

        assert count == 3
        assert index.size == 3
        _, ids = index.search(records[1].vector.reshape(1, -1), top_k=1)
        assert ids[0][0] == records[1].vector_id

    def test_rebuild_discards_previous_contents(
        self, repo: PersonRepository, index: FaissIndex
    ) -> None:
        index.add([999], unit_vectors(1, seed=9))
        person = repo.create("Hamza")
        repo.add_embeddings(person.id, unit_vectors(2))

        index.rebuild_from(repo.list_embeddings())

        _, ids = index.search(unit_vectors(1, seed=9), top_k=3)
        assert 999 not in ids[0].tolist()

    def test_rebuild_from_nothing_yields_an_empty_index(self, index: FaissIndex) -> None:
        index.add([1], unit_vectors(1))

        assert index.rebuild_from([]) == 0
        assert index.size == 0

    def test_rebuild_rejects_mismatched_dimensions(
        self, repo: PersonRepository, tmp_path: Path
    ) -> None:
        person = repo.create("Hamza")
        repo.add_embeddings(person.id, unit_vectors(1))
        wrong = FaissIndex(tmp_path / "wrong.faiss", dim=DIM * 2)

        with pytest.raises(VectorIndexError, match="do not match the configured dimension"):
            wrong.rebuild_from(repo.list_embeddings())


class TestBlobStore:
    def test_save_and_read(self, blobs: BlobStore) -> None:
        relative = blobs.save("person-1", "01.jpg", b"jpeg-bytes")

        assert relative == "person-1/01.jpg"
        assert blobs.read(relative) == b"jpeg-bytes"
        assert blobs.exists(relative)

    def test_directory_traversal_in_the_filename_is_stripped(self, blobs: BlobStore) -> None:
        """A crafted upload name must not be able to write outside the store."""
        relative = blobs.save("person-1", "../../evil.jpg", b"x")

        assert relative == "person-1/evil.jpg"
        assert blobs.resolve(relative).parent.name == "person-1"

    def test_traversal_in_a_stored_path_is_rejected(self, blobs: BlobStore) -> None:
        with pytest.raises(StorageError, match="escapes the blob directory"):
            blobs.resolve("../../../etc/passwd")

    def test_empty_data_is_rejected(self, blobs: BlobStore) -> None:
        with pytest.raises(StorageError, match="empty image"):
            blobs.save("person-1", "01.jpg", b"")

    def test_reading_a_missing_file_raises(self, blobs: BlobStore) -> None:
        with pytest.raises(StorageError, match="Could not read"):
            blobs.read("person-1/absent.jpg")

    def test_delete_person_removes_every_file(self, blobs: BlobStore) -> None:
        blobs.save("person-1", "01.jpg", b"a")
        blobs.save("person-1", "02.jpg", b"b")

        assert blobs.delete_person("person-1") == 2
        assert blobs.delete_person("person-1") == 0
