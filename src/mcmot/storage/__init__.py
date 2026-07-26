"""Persistence: relational rows, gallery vectors, and image files.

Three backends behind one boundary, with a strict ownership rule:

**SQLite is the source of truth. The vector index is a derived cache.**

The two stores have no shared transaction, so they can diverge — a crash between
"person deleted" and "vectors removed" would leave an index entry pointing at
nobody. Rather than pretend otherwise, divergence is made cheap to detect (compare
counts) and cheap to fix (:meth:`~mcmot.storage.vector_index.FaissIndex.rebuild_from`),
and the index holds nothing that the database cannot reconstruct.
"""

from __future__ import annotations

from mcmot.storage.blobs import BlobStore
from mcmot.storage.db import Database
from mcmot.storage.models import Base, EmbeddingRow, PersonRow
from mcmot.storage.repository import PersonAlreadyExistsError, PersonRepository
from mcmot.storage.vector_index import FaissIndex

__all__ = [
    "Base",
    "BlobStore",
    "Database",
    "EmbeddingRow",
    "FaissIndex",
    "PersonAlreadyExistsError",
    "PersonRepository",
    "PersonRow",
]
