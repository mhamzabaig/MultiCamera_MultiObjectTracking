"""Database engine, session management, and SQLite tuning.

Four PRAGMAs are set on every connection, and each one prevents a real failure:

``foreign_keys=ON``
    SQLite ignores foreign keys unless asked. Without this, deleting a person
    silently orphans their embeddings and the ``ON DELETE CASCADE`` in the schema
    does nothing.
``journal_mode=WAL``
    Readers no longer block the writer. Several camera pipeline threads read the
    gallery while registration writes to it; without WAL they serialise and stall.
``synchronous=NORMAL``
    With WAL this is durable against process crashes (only a power loss can lose
    the last commits), and avoids an fsync per transaction.
``busy_timeout``
    Wait for a held write lock instead of failing instantly with "database is
    locked" — the single most common SQLite error under concurrency.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from mcmot.core.errors import StorageError
from mcmot.core.logging import get_logger
from mcmot.core.settings import StorageSettings
from mcmot.storage.models import Base

__all__ = ["Database"]

_log = get_logger(__name__)

_SQLITE_PREFIX = "sqlite:///"
_BUSY_TIMEOUT_MS = 5000


class Database:
    """Owns the engine and hands out sessions."""

    def __init__(self, config: StorageSettings) -> None:
        """Create the engine.

        Args:
            config: Validated storage configuration.

        Raises:
            StorageError: If the database directory cannot be created or the
                engine cannot be constructed.
        """
        self._config = config
        self._url = config.database_url
        self._is_sqlite = self._url.startswith("sqlite")
        self._engine = self._create_engine()
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False)

    # -- construction -----------------------------------------------------
    def _create_engine(self) -> Engine:
        connect_args: dict[str, Any] = {}
        if self._is_sqlite:
            self._ensure_parent_directory()
            # Pipeline threads share this engine; SQLite's default same-thread
            # guard would reject their connections.
            connect_args["check_same_thread"] = False

        try:
            engine = create_engine(
                self._url,
                echo=self._config.echo_sql,
                future=True,
                connect_args=connect_args,
            )
        except SQLAlchemyError as exc:
            raise StorageError(
                "Could not create the database engine.",
                details={"url": self._url, "reason": str(exc)},
            ) from exc

        if self._is_sqlite:
            self._install_sqlite_pragmas(engine)
        return engine

    def _ensure_parent_directory(self) -> None:
        path = self.file_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError(
                f"Cannot create the database directory: {path.parent}",
                details={"path": str(path.parent), "reason": str(exc)},
            ) from exc

    def _install_sqlite_pragmas(self, engine: Engine) -> None:
        wal = self._config.sqlite_wal

        @event.listens_for(engine, "connect")
        def _set_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                if wal:
                    cursor.execute("PRAGMA journal_mode=WAL")
                    cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            finally:
                cursor.close()

    # -- accessors --------------------------------------------------------
    @property
    def engine(self) -> Engine:
        """The underlying SQLAlchemy engine."""
        return self._engine

    @property
    def url(self) -> str:
        """The configured database URL."""
        return self._url

    @property
    def file_path(self) -> Path | None:
        """Filesystem path of the SQLite file, or ``None`` for other backends."""
        if not self._url.startswith(_SQLITE_PREFIX):
            return None
        return Path(self._url[len(_SQLITE_PREFIX) :])

    # -- lifecycle --------------------------------------------------------
    def create_schema(self) -> None:
        """Create any missing tables.

        Sufficient while the schema has never shipped. Once a deployment holds
        real data and the schema must change, this is the point at which a
        migration tool earns its keep.
        """
        try:
            Base.metadata.create_all(self._engine)
        except SQLAlchemyError as exc:
            raise StorageError(
                "Could not create the database schema.",
                details={"url": self._url, "reason": str(exc)},
            ) from exc
        _log.info("storage.schema_ready", url=self._url)

    def dispose(self) -> None:
        """Close pooled connections. Safe to call more than once."""
        self._engine.dispose()

    # -- sessions ---------------------------------------------------------
    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield a session, committing on success and rolling back on failure.

        Every database error is translated into :class:`StorageError` here so
        callers never have to import SQLAlchemy exceptions, and no partial write
        can survive an exception.
        """
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            raise StorageError("Database operation failed.", details={"reason": str(exc)}) from exc
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
