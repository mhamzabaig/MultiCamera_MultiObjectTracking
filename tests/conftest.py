"""Shared fixtures.

Two invariants every test in this suite relies on:

* **The developer's environment cannot influence a test.** ``MCMOT*`` variables
  are stripped and the dotenv path is pointed at a file that does not exist, so a
  real ``.env`` on this machine can never change an assertion.
* **Nothing is written inside the repository.** Data, model, and log directories
  are redirected under ``tmp_path``, so a test run leaves the working tree clean.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from mcmot.core.settings import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Remove every MCMOT variable and disable dotenv discovery."""
    for name in list(os.environ):
        if name.upper().startswith("MCMOT"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MCMOT_ENV_FILE", str(tmp_path / "absent.env"))


@pytest.fixture
def repo_config_dir() -> Path:
    """The committed ``config/`` directory, for tests asserting shipped defaults."""
    return REPO_CONFIG_DIR


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """An empty config directory: every value comes from code defaults."""
    path = tmp_path / "config"
    path.mkdir()
    return path


@pytest.fixture
def settings(tmp_path: Path, config_dir: Path) -> Settings:
    """Settings backed by throwaway directories and code defaults."""
    return load_settings(
        config_dir=config_dir,
        env="test",
        paths={"base_dir": str(tmp_path / "workspace")},
    )


@pytest.fixture
def write_yaml() -> Iterator[object]:
    """Helper writing a YAML file from raw text."""

    def _write(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    yield _write
