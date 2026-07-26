"""Container lifecycle tests."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import structlog

from mcmot.container import Container
from mcmot.core.device import DeviceManager
from mcmot.core.errors import ConfigError, DeviceError
from mcmot.core.settings import Settings, load_settings


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a machine without PyTorch, forcing the CPU fallback path."""
    monkeypatch.setattr(DeviceManager, "_import_torch", staticmethod(lambda: None))


def build_container(config_dir: Path, tmp_path: Path, **overrides) -> Container:
    """Build a container whose whole filesystem footprint is inside tmp_path."""
    return Container(
        load_settings(
            config_dir=config_dir,
            env="test",
            paths={"base_dir": str(tmp_path / "workspace")},
            **overrides,
        )
    )


@pytest.fixture
def container(config_dir: Path, tmp_path: Path) -> Container:
    """A container pinned to CPU, so no test depends on the host GPU."""
    return build_container(config_dir, tmp_path, device={"prefer": "cpu"})


class TestStartup:
    def test_startup_creates_directories(self, container: Container) -> None:
        container.startup()

        paths = container.settings.paths
        assert paths.data_dir.is_dir()
        assert paths.models_dir.is_dir()
        assert paths.logs_dir.is_dir()

    def test_startup_writes_a_boot_log(self, container: Container) -> None:
        container.startup()

        content = container.settings.log_file_path.read_text(encoding="utf-8")
        assert "app.starting" in content
        assert "app.started" in content

    def test_boot_log_records_the_effective_configuration(self, container: Container) -> None:
        """The single most useful line in a post-mortem: what was actually running."""
        container.startup()

        content = container.settings.log_file_path.read_text(encoding="utf-8")
        assert "tau_accept" in content
        assert "yolo11n.pt" in content

    def test_startup_resolves_the_device_eagerly(self, container: Container) -> None:
        """Deferring CUDA probing to a worker thread would race; boot resolves it."""
        container.startup()

        assert container.device_manager.info.type == "cpu"

    def test_uptime_is_zero_before_startup(self, container: Container) -> None:
        assert container.uptime_s == 0.0
        assert container.started_epoch is None

    def test_uptime_advances_after_startup(self, container: Container) -> None:
        container.startup()

        assert container.uptime_s >= 0.0
        assert container.started_epoch is not None

    def test_device_manager_is_a_singleton(self, container: Container) -> None:
        assert container.device_manager is container.device_manager

    def test_startup_propagates_device_errors(
        self, config_dir: Path, tmp_path: Path, no_torch: None
    ) -> None:
        """A demanded-but-absent GPU must fail the boot, not the first frame."""
        container = build_container(
            config_dir, tmp_path, device={"prefer": "cuda", "allow_cpu_fallback": False}
        )

        with pytest.raises(DeviceError):
            container.startup()

    def test_unwritable_directory_is_reported_as_a_config_error(
        self, settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def deny(*args: object, **kwargs: object) -> None:
            raise OSError("access denied")

        monkeypatch.setattr(Path, "mkdir", deny)

        with pytest.raises(ConfigError, match="Cannot create required directory"):
            Container(settings).startup()

    def test_container_loads_settings_when_none_supplied(
        self, monkeypatch: pytest.MonkeyPatch, config_dir: Path
    ) -> None:
        monkeypatch.setenv("MCMOT_CONFIG_DIR", str(config_dir))

        assert Container().settings.identity.tau_accept == 0.55


class TestHealth:
    def test_health_reports_ok_on_a_clean_resolution(self, container: Container) -> None:
        container.startup()

        health = container.health()

        assert health["status"] == "ok"
        assert health["env"] == "test"
        assert health["device"]["type"] == "cpu"
        assert "identity" in health["config"]

    def test_health_reports_degraded_on_a_fallback(
        self, config_dir: Path, tmp_path: Path, no_torch: None
    ) -> None:
        """A CPU fallback is a working system; a monitor must be able to tell."""
        container = build_container(config_dir, tmp_path)
        container.startup()

        assert container.health()["status"] == "degraded"
        assert container.health()["device"]["is_fallback"] is True

    def test_health_is_available_before_startup(self, container: Container) -> None:
        assert container.health()["uptime_s"] == 0.0

    def test_health_redacts_database_credentials(self, config_dir: Path, tmp_path: Path) -> None:
        container = build_container(
            config_dir,
            tmp_path,
            device={"prefer": "cpu"},
            storage={"database_url": "postgresql://u:p@host/db"},
        )

        assert "p@host" not in container.health()["config"]["storage"]["database"]


class TestStorageWiring:
    """The DB<->index consistency contract, exercised through real startup."""

    @staticmethod
    def unit_vectors(count: int, dim: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        raw = rng.normal(size=(count, dim)).astype(np.float32)
        return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)

    def test_startup_creates_schema_and_index(self, container: Container) -> None:
        container.startup()

        assert container.person_repository.count() == 0
        assert container.vector_index.size == 0

    def test_shutdown_persists_the_index(self, container: Container) -> None:
        container.startup()
        person = container.person_repository.create("Hamza")
        dim = container.settings.reid.embedding_dim
        records = container.person_repository.add_embeddings(person.id, self.unit_vectors(2, dim))
        container.vector_index.add([r.vector_id for r in records], self.unit_vectors(2, dim))

        container.shutdown()

        assert container.settings.vector_index_file.is_file()

    def test_divergent_index_is_rebuilt_on_startup(self, config_dir: Path, tmp_path: Path) -> None:
        """A crash between deleting a person and dropping their vectors must heal.

        Simulated by writing embeddings to the database while leaving the index
        file empty, which is exactly what that crash window looks like on disk.
        """
        first = build_container(config_dir, tmp_path, device={"prefer": "cpu"})
        first.startup()
        dim = first.settings.reid.embedding_dim
        person = first.person_repository.create("Hamza")
        first.person_repository.add_embeddings(person.id, self.unit_vectors(3, dim))
        first.vector_index.save()  # saved while still empty -> divergence
        first.shutdown()

        second = build_container(config_dir, tmp_path, device={"prefer": "cpu"})
        second.startup()

        assert second.vector_index.size == 3
        assert second.person_repository.count_embeddings() == 3
        second.shutdown()

    def test_rebuild_can_be_disabled(self, config_dir: Path, tmp_path: Path) -> None:
        """With rebuilding off, divergence is reported and left for a human."""
        first = build_container(config_dir, tmp_path, device={"prefer": "cpu"})
        first.startup()
        dim = first.settings.reid.embedding_dim
        person = first.person_repository.create("Hamza")
        first.person_repository.add_embeddings(person.id, self.unit_vectors(3, dim))
        first.vector_index.save()
        first.shutdown()

        second = build_container(
            config_dir,
            tmp_path,
            device={"prefer": "cpu"},
            storage={"rebuild_index_on_mismatch": False},
        )
        second.startup()

        assert second.vector_index.size == 0
        content = second.settings.log_file_path.read_text(encoding="utf-8")
        assert "index.rebuild_disabled" in content
        second.shutdown()


class TestShutdown:
    def test_shutdown_logs_completion(self, container: Container) -> None:
        container.startup()

        container.shutdown()

        content = container.settings.log_file_path.read_text(encoding="utf-8")
        assert "app.stopped" in content

    def test_shutdown_is_safe_without_startup(self, container: Container) -> None:
        """FastAPI runs lifespan shutdown even when startup raised part-way."""
        container.shutdown()

    def test_shutdown_is_idempotent(self, container: Container) -> None:
        container.startup()

        container.shutdown()
        container.shutdown()
