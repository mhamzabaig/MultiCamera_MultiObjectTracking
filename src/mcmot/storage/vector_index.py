"""FAISS gallery index.

``IndexIDMap2`` wrapping ``IndexFlatIP``:

* **FlatIP** is exhaustive, and deliberately so. It is SIMD/BLAS-vectorised, not a
  Python loop, and at gallery scale (hundreds to low thousands of vectors) a search
  costs microseconds. An approximate index would trade that for recall loss, and a
  missed nearest neighbour here means a wrong name drawn on a person.
* **Inner product on L2-normalised vectors is exactly cosine similarity**, so
  scores are interpretable in ``[-1, 1]`` and thresholds survive a model swap. The
  normalisation invariant is enforced on insert — silently accepting an
  unnormalised vector would make every score meaningless.
* **IDMap2** supplies the stable external ids, ``remove_ids``, and reconstruction
  that plain ``IndexFlat`` lacks.

The index is a *derived cache*. The database is the source of truth, so anything
that goes wrong here is recoverable with :meth:`FaissIndex.rebuild_from`.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

import faiss
import numpy as np

from mcmot.core.errors import VectorIndexError
from mcmot.core.logging import get_logger
from mcmot.domain.person import EmbeddingRecord
from mcmot.domain.types import EmbeddingMatrix, IdArray, ScoreArray

__all__ = ["FaissIndex"]

_log = get_logger(__name__)

#: Tolerance for the unit-norm check. Generous enough for float32 rounding after
#: a normalise-then-serialise round trip, tight enough to catch a raw vector.
_NORM_TOLERANCE = 1e-3


class FaissIndex:
    """A persistent, thread-safe exact-cosine index over gallery embeddings."""

    def __init__(self, path: Path, dim: int) -> None:
        """Create an empty in-memory index.

        Call :meth:`load` to read a persisted one from ``path``.

        Args:
            path: File the index is saved to and loaded from.
            dim: Embedding dimensionality, from ``reid.embedding_dim``.
        """
        if dim <= 0:
            raise VectorIndexError("Embedding dimension must be positive.", details={"dim": dim})
        self._path = path
        self._dim = dim
        # One lock for the whole index: FAISS objects are not safe under
        # concurrent mutation, and several camera threads will search while
        # registration writes.
        self._lock = threading.RLock()
        self._index = self._new_index()

    def _new_index(self) -> faiss.Index:
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dim))

    # -- properties -------------------------------------------------------
    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    @property
    def path(self) -> Path:
        """File the index persists to."""
        return self._path

    @property
    def size(self) -> int:
        """Number of vectors currently indexed."""
        with self._lock:
            return int(self._index.ntotal)

    # -- mutation ---------------------------------------------------------
    def add(self, vector_ids: Sequence[int], vectors: EmbeddingMatrix) -> None:
        """Insert vectors under the given ids.

        Args:
            vector_ids: One id per row, from ``embeddings.vector_id``.
            vectors: ``float32`` array of shape ``(n, dim)``, L2-normalised.

        Raises:
            VectorIndexError: On a shape, dtype, id-count, or normalisation error.
        """
        matrix = self._prepare(vectors)
        if len(vector_ids) != matrix.shape[0]:
            raise VectorIndexError(
                "One id is required per vector.",
                details={"ids": len(vector_ids), "vectors": int(matrix.shape[0])},
            )
        ids = np.asarray(vector_ids, dtype=np.int64)
        with self._lock:
            self._index.add_with_ids(matrix, ids)

    def remove(self, vector_ids: Sequence[int]) -> int:
        """Remove vectors by id.

        Returns:
            The number of vectors actually removed. Ids that are absent are
            ignored, which makes the call idempotent — important because deletion
            is a two-store operation that may be retried.
        """
        if not vector_ids:
            return 0
        ids = np.asarray(vector_ids, dtype=np.int64)
        with self._lock:
            return int(self._index.remove_ids(ids))

    def reset(self) -> None:
        """Discard every vector, keeping the configured dimensionality."""
        with self._lock:
            self._index = self._new_index()

    def rebuild_from(self, records: Iterable[EmbeddingRecord]) -> int:
        """Replace the index contents from the database.

        The recovery path for any divergence: the index holds nothing that cannot
        be reconstructed from ``embeddings``.

        Args:
            records: Every embedding to index.

        Returns:
            The number of vectors indexed.

        Raises:
            VectorIndexError: If a record's dimensionality does not match.
        """
        items = list(records)
        with self._lock:
            self._index = self._new_index()
            if not items:
                _log.info("index.rebuilt", vectors=0)
                return 0

            mismatched = [r.vector_id for r in items if r.dim != self._dim]
            if mismatched:
                raise VectorIndexError(
                    "Stored embeddings do not match the configured dimension; "
                    "the ReID model or its configuration has changed.",
                    details={
                        "expected_dim": self._dim,
                        "vector_ids": mismatched[:10],
                        "mismatched": len(mismatched),
                    },
                )

            matrix = np.ascontiguousarray(
                np.stack([r.vector for r in items]).astype(np.float32, copy=False)
            )
            ids = np.asarray([r.vector_id for r in items], dtype=np.int64)
            self._index.add_with_ids(matrix, ids)

        _log.info("index.rebuilt", vectors=len(items))
        return len(items)

    # -- query ------------------------------------------------------------
    def search(self, queries: EmbeddingMatrix, top_k: int) -> tuple[ScoreArray, IdArray]:
        """Find the nearest gallery vectors for each query.

        Args:
            queries: ``float32`` array of shape ``(n, dim)``, L2-normalised.
            top_k: Neighbours to return per query.

        Returns:
            ``(scores, ids)``, each of shape ``(n, top_k)``. Where fewer than
            ``top_k`` vectors exist, FAISS pads ids with ``-1`` and scores with a
            large negative value; callers must filter on ``id >= 0``.

        Raises:
            VectorIndexError: On an invalid query array or non-positive ``top_k``.
        """
        if top_k <= 0:
            raise VectorIndexError("top_k must be positive.", details={"top_k": top_k})
        matrix = self._prepare(queries)
        with self._lock:
            if self._index.ntotal == 0:
                # FAISS handles this, but returning early keeps the empty-gallery
                # case explicit and allocation-free.
                empty_scores = np.full((matrix.shape[0], top_k), -1.0, dtype=np.float32)
                empty_ids = np.full((matrix.shape[0], top_k), -1, dtype=np.int64)
                return empty_scores, empty_ids
            scores, ids = self._index.search(matrix, top_k)
        return scores.astype(np.float32, copy=False), ids

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        """Write the index to disk atomically.

        Written to a temporary file and renamed, so a crash mid-write leaves the
        previous index intact rather than a truncated one. ``Path.replace`` is
        atomic within a filesystem, including on Windows.

        Raises:
            VectorIndexError: If the file cannot be written.
        """
        temp = self._path.with_suffix(self._path.suffix + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                faiss.write_index(self._index, str(temp))
                count = int(self._index.ntotal)
            temp.replace(self._path)
        except (OSError, RuntimeError) as exc:
            temp.unlink(missing_ok=True)
            raise VectorIndexError(
                f"Could not write the vector index: {self._path}",
                details={"path": str(self._path), "reason": str(exc)},
            ) from exc
        _log.info("index.saved", path=str(self._path), vectors=count)

    def load(self) -> bool:
        """Read the index from disk if it exists.

        Returns:
            ``True`` if an index was loaded, ``False`` if no file was present (in
            which case the index stays empty and can be rebuilt).

        Raises:
            VectorIndexError: If the file exists but is unreadable, or was built
                for a different dimensionality.
        """
        if not self._path.is_file():
            return False
        try:
            index = faiss.read_index(str(self._path))
        except (OSError, RuntimeError) as exc:
            raise VectorIndexError(
                f"Could not read the vector index: {self._path}. "
                "Delete the file and rebuild it from the database.",
                details={"path": str(self._path), "reason": str(exc)},
            ) from exc

        if int(index.d) != self._dim:
            raise VectorIndexError(
                "The persisted index was built for a different embedding dimension.",
                details={"path": str(self._path), "found": int(index.d), "expected": self._dim},
            )

        with self._lock:
            self._index = index
        _log.info("index.loaded", path=str(self._path), vectors=int(index.ntotal))
        return True

    # -- helpers ----------------------------------------------------------
    def _prepare(self, vectors: EmbeddingMatrix) -> EmbeddingMatrix:
        """Validate shape, dtype, and unit norm; return a contiguous array."""
        if vectors.ndim != 2:
            raise VectorIndexError(
                "Vectors must be a 2-D (n, dim) array.", details={"shape": list(vectors.shape)}
            )
        if vectors.shape[0] == 0:
            raise VectorIndexError("No vectors supplied.", details={"shape": list(vectors.shape)})
        if int(vectors.shape[1]) != self._dim:
            raise VectorIndexError(
                "Vector dimension does not match the index.",
                details={"expected": self._dim, "found": int(vectors.shape[1])},
            )
        if vectors.dtype != np.float32:
            raise VectorIndexError(
                "Vectors must be float32.", details={"dtype": str(vectors.dtype)}
            )

        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=_NORM_TOLERANCE):
            worst = int(np.argmax(np.abs(norms - 1.0)))
            raise VectorIndexError(
                "Vectors must be L2-normalised; inner-product search only equals "
                "cosine similarity for unit vectors.",
                details={"row": worst, "norm": round(float(norms[worst]), 6)},
            )
        return np.ascontiguousarray(vectors)
