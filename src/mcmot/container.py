"""Application container: composition root and dependency provider.

Deliberately at the package root rather than inside :mod:`mcmot.core`. The container
imports from every feature package — storage, detection, tracking, reid — while those
packages import from ``core``. Placing it in ``core`` created an import cycle that
only manifested depending on which module a process imported first: ``core.device``
would trigger ``core/__init__``, which imported the container, which imported
``reid``, which imported ``core.device`` again. The composition root belongs *above*
the layers it wires together, not inside the lowest one.

Why a hand-written container rather than a DI framework: the object graph here is
small and known at build time. A framework would add magic, a decorator
vocabulary, and a debugging surface for no benefit. What we need is one place
that knows how to construct things, and one place that owns their lifetime.

Two rules the rest of the codebase follows:

* **No module-level singletons.** Nothing does ``settings = load_settings()`` at
  import time. Every collaborator arrives through a constructor argument, which
  is exactly what makes the fake detector / in-memory database substitutions in
  later phases possible without monkeypatching.
* **Construction is explicit, wiring is lazy.** Components are built on first
  access, but :meth:`Container.startup` materialises them eagerly during boot so
  a misconfiguration fails at startup rather than on the first tracked frame.

Later phases attach their singletons here — repositories and the vector index in
phase 2, the model registry in phases 3–5, the identity service in phase 6, and
the camera manager in phase 7 — using the same pattern.
"""

from __future__ import annotations

import time
from functools import cached_property
from typing import Any

from mcmot.core.device import DeviceManager
from mcmot.core.logging import configure_logging, get_logger
from mcmot.core.settings import Settings, load_settings
from mcmot.detection import BaseDetector, create_detector
from mcmot.identity import IdentityService
from mcmot.reid import BaseEmbedder, QualityGate, create_embedder
from mcmot.storage.blobs import BlobStore
from mcmot.storage.db import Database
from mcmot.storage.repository import PersonRepository
from mcmot.storage.vector_index import FaissIndex
from mcmot.tracking import BaseTracker, create_tracker

__all__ = ["Container"]


class Container:
    """Owns construction and lifetime of the application's long-lived objects."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create the container.

        Args:
            settings: Pre-built configuration; loaded from the standard layers
                when omitted.
        """
        self._settings = settings if settings is not None else load_settings()
        self._started_monotonic: float | None = None
        self._started_epoch: float | None = None
        self._log = get_logger(__name__)

    # -- provided components ----------------------------------------------
    @property
    def settings(self) -> Settings:
        """The validated application configuration."""
        return self._settings

    @cached_property
    def device_manager(self) -> DeviceManager:
        """Compute device resolver, shared by every model in the process."""
        return DeviceManager(self._settings.device)

    @cached_property
    def database(self) -> Database:
        """Relational store: the source of truth for identities and embeddings."""
        return Database(self._settings.storage)

    @cached_property
    def person_repository(self) -> PersonRepository:
        """Reads and writes enrolled people and their gallery embeddings."""
        return PersonRepository(self.database)

    @cached_property
    def vector_index(self) -> FaissIndex:
        """Gallery similarity index, derived from the database."""
        return FaissIndex(self._settings.vector_index_file, dim=self._settings.reid.embedding_dim)

    @cached_property
    def blob_store(self) -> BlobStore:
        """Person image files."""
        return BlobStore(self._settings.blob_directory)

    @cached_property
    def detector(self) -> BaseDetector:
        """Person detector, shared by every camera pipeline.

        One instance for the whole process: a per-camera copy would exhaust 6 GB of
        VRAM at three cameras and triple warm-up time. Built lazily so that a
        machine without model weights can still run the API and the tests.
        """
        detector = create_detector(self._settings.detector, self.device_manager)
        detector.warmup()
        return detector

    @cached_property
    def embedder(self) -> BaseEmbedder:
        """ReID embedder, shared by every camera pipeline.

        Safe to share: unlike the tracker it holds no per-stream state. Built lazily
        so a machine without model weights can still run the API and the tests.
        """
        embedder = create_embedder(self._settings.reid, self.device_manager)
        embedder.warmup()
        return embedder

    @cached_property
    def quality_gate(self) -> QualityGate:
        """Crop quality gate, shared by registration and tracking alike."""
        return QualityGate(self._settings.reid.quality)

    @cached_property
    def identity_service(self) -> IdentityService:
        """Enrolment and identity resolution, shared by every camera.

        Deliberately one instance: it holds the per-track vote state that makes
        identities stable, so a second instance would split that evidence and each
        camera would start from scratch.
        """
        return IdentityService(
            config=self._settings.identity,
            embedder=self.embedder,
            quality_gate=self.quality_gate,
            repository=self.person_repository,
            index=self.vector_index,
            blobs=self.blob_store,
        )

    def create_tracker(self) -> BaseTracker:
        """Build a tracker for one camera.

        A method, not a cached property: trackers hold per-track Kalman state, so
        each camera needs its own instance. Sharing one would associate people
        across unrelated views.
        """
        return create_tracker(self._settings.tracker)

    # -- lifecycle --------------------------------------------------------
    def startup(self) -> None:
        """Prepare the process: directories, logging, device, then eager wiring.

        Ordering matters. Directories come first because the log file lives in
        one of them; logging comes second so that device resolution — the first
        step that can legitimately warn — is captured; components come last so
        their construction is visible in the log.

        Raises:
            ConfigError: If a required directory or the log file is unwritable.
            DeviceError: If CUDA was demanded and fallback is forbidden.
        """
        settings = self._settings
        settings.paths.ensure_directories()

        configure_logging(settings.logging, log_dir=settings.paths.logs_dir, force=True)
        self._log = get_logger(__name__)

        self._started_monotonic = time.monotonic()
        self._started_epoch = time.time()

        self._log.info(
            "app.starting",
            version=settings.version,
            env=settings.env,
            base_dir=str(settings.paths.base_dir),
            config=settings.summary(),
        )

        # Resolve eagerly: probing CUDA costs ~0.5 s on first touch, and a
        # cached_property racing between camera worker threads is a hazard worth
        # avoiding entirely rather than reasoning about.
        device = self.device_manager.info

        self.database.create_schema()
        self._prepare_vector_index()

        self._log.info(
            "app.started",
            device=device.torch_device,
            fp16=device.fp16,
            persons=self.person_repository.count(),
            embeddings=self.vector_index.size,
        )

    def _prepare_vector_index(self) -> None:
        """Load the index and reconcile it with the database.

        The two stores share no transaction, so a crash between deleting a person
        and removing their vectors can leave the index holding entries that belong
        to nobody. Comparing sizes catches that in microseconds, and because the
        index is a derived cache, rebuilding from the database always resolves it.

        With ``storage.rebuild_index_on_mismatch`` disabled the divergence is
        logged as an error and left alone — appropriate when a human should inspect
        it first.
        """
        index = self.vector_index
        loaded = index.load()
        expected = self.person_repository.count_embeddings()

        if not loaded and expected == 0:
            return  # fresh installation, nothing to reconcile

        if index.size == expected:
            self._log.info("index.ready", vectors=index.size, loaded_from_disk=loaded)
            return

        self._log.warning(
            "index.mismatch",
            index_vectors=index.size,
            database_embeddings=expected,
            loaded_from_disk=loaded,
        )
        if not self._settings.storage.rebuild_index_on_mismatch:
            self._log.error("index.rebuild_disabled", action="manual intervention required")
            return

        index.rebuild_from(self.person_repository.list_embeddings())
        index.save()

    def shutdown(self) -> None:
        """Release resources in reverse order of acquisition.

        Idempotent, and safe to call after a failed startup: FastAPI's lifespan
        shutdown runs even when startup raised part-way through.
        """
        self._log.info("app.stopping", uptime_s=round(self.uptime_s, 2))

        # Reverse construction order. Each step is guarded: a failure while
        # shutting down must not prevent the remaining resources from being
        # released, and shutdown also runs after a partially-failed startup.
        if "vector_index" in self.__dict__:
            try:
                self.vector_index.save()
            except Exception:  # noqa: BLE001 - never let cleanup mask the shutdown
                self._log.exception("index.save_failed_on_shutdown")

        if "database" in self.__dict__:
            try:
                self.database.dispose()
            except Exception:  # noqa: BLE001
                self._log.exception("database.dispose_failed")

        # Phase 7+ stops camera workers before these, once they exist.
        self._log.info("app.stopped")

    # -- introspection ----------------------------------------------------
    @property
    def uptime_s(self) -> float:
        """Seconds since :meth:`startup`; ``0.0`` if it has not run."""
        if self._started_monotonic is None:
            return 0.0
        return time.monotonic() - self._started_monotonic

    @property
    def started_epoch(self) -> float | None:
        """Wall-clock start time as a UNIX epoch, or ``None`` before startup."""
        return self._started_epoch

    def health(self) -> dict[str, Any]:
        """Return a health snapshot for ``/api/system/health``.

        Reports ``degraded`` rather than ``unhealthy`` when running on a CPU
        fallback: the system is genuinely serving requests, just slowly, and a
        monitor should be able to tell those two situations apart.
        """
        device = self.device_manager.info
        status = "degraded" if device.is_fallback else "ok"
        return {
            "status": status,
            "version": self._settings.version,
            "env": self._settings.env,
            "uptime_s": round(self.uptime_s, 2),
            "started_epoch": self._started_epoch,
            "device": device.to_dict(),
            "config": self._settings.summary(),
        }
