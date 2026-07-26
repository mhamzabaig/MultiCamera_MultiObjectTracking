"""Create the database schema and an empty vector index.

Idempotent: safe to run against an existing installation, where it reports what is
already there instead of changing anything.

    python scripts/init_db.py
    python scripts/init_db.py --rebuild-index    # discard and rebuild from the database
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mcmot.core.logging import configure_logging  # noqa: E402
from mcmot.core.settings import load_settings  # noqa: E402
from mcmot.storage.blobs import BlobStore  # noqa: E402
from mcmot.storage.db import Database  # noqa: E402
from mcmot.storage.repository import PersonRepository  # noqa: E402
from mcmot.storage.vector_index import FaissIndex  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialise MC-MOT storage.")
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the vector index from the database, discarding the file on disk.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    settings.paths.ensure_directories()
    configure_logging(settings.logging, log_dir=settings.paths.logs_dir, force=True)

    database = Database(settings.storage)
    database.create_schema()
    repository = PersonRepository(database)

    index = FaissIndex(settings.vector_index_file, dim=settings.reid.embedding_dim)
    if args.rebuild_index:
        count = index.rebuild_from(repository.list_embeddings())
        index.save()
        print(f"rebuilt index with {count} vector(s)")
    elif not index.load():
        index.save()
        print("created an empty index")

    BlobStore(settings.blob_directory).root.mkdir(parents=True, exist_ok=True)

    persons = repository.count()
    embeddings = repository.count_embeddings()

    print()
    print(f"database   {database.url}")
    print(f"index      {index.path}  ({index.size} vector(s), dim {index.dim})")
    print(f"images     {settings.blob_directory}")
    print(f"enrolled   {persons} person(s), {embeddings} embedding(s)")
    if index.size != embeddings:
        print()
        print(
            f"WARNING: index holds {index.size} vector(s) but the database has "
            f"{embeddings}. Run with --rebuild-index."
        )
        database.dispose()
        return 1

    database.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
