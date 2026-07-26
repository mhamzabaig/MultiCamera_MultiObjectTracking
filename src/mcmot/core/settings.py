"""Layered, validated application configuration.

Resolution order (lowest priority first)::

    field defaults in this module
      -> config/default.yaml          (committed, operational baseline)
      -> config/local.yaml            (gitignored, machine-specific)
      -> .env                         (gitignored)
      -> process environment          (MCMOT__SECTION__KEY)
      -> explicit keyword arguments   (tests, scripts)

Sources are *deep merged*, so ``config/local.yaml`` may override
``identity.tau_accept`` without restating the rest of the ``identity`` block.

Two deliberate design choices are worth calling out.

**Typos are fatal.** Every model sets ``extra="forbid"``. A misspelled key in
YAML or a stray ``MCMOT__`` environment variable aborts startup with a precise
message rather than being silently dropped — a silently ignored threshold is a
bug that surfaces weeks later as a wrong name drawn on someone's head.

**Validation happens exactly once, at boot.** Nothing in the hot path re-reads
configuration or re-checks bounds. Settings are constructed, validated, logged,
and thereafter treated as immutable and injected through the DI container.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from mcmot import __version__
from mcmot.core.errors import ConfigError

__all__ = [
    "ApiSettings",
    "CameraSettings",
    "DetectorSettings",
    "DeviceSettings",
    "IdentitySettings",
    "LoggingSettings",
    "PathsSettings",
    "PipelineSettings",
    "QualitySettings",
    "ReIDSettings",
    "Settings",
    "StorageSettings",
    "TrackerSettings",
    "load_settings",
    "resolve_config_dir",
]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# ---------------------------------------------------------------------------
# Project root / config directory discovery
# ---------------------------------------------------------------------------
_config_dir_override: ContextVar[Path | None] = ContextVar(
    "mcmot_config_dir_override", default=None
)


def _detect_project_root() -> Path:
    """Locate the project root.

    Walks upward looking for ``pyproject.toml``, which works for both an
    editable install (``src`` layout) and a plain ``PYTHONPATH`` run. Falls back
    to a fixed depth for wheel installs where no marker file exists; in that
    case ``MCMOT_PROJECT_ROOT`` should be set explicitly.
    """
    override = os.getenv("MCMOT_PROJECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    # src/mcmot/core/settings.py -> MTMCT
    return here.parents[3]


def resolve_config_dir() -> Path:
    """Return the directory holding ``default.yaml`` and ``local.yaml``."""
    override = _config_dir_override.get()
    if override is not None:
        return override
    env = os.getenv("MCMOT_CONFIG_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _detect_project_root() / "config"


@contextmanager
def _config_dir_context(path: Path) -> Iterator[None]:
    """Temporarily redirect config discovery, used by tests and scripts."""
    token = _config_dir_override.set(path)
    try:
        yield
    finally:
        _config_dir_override.reset(token)


# ---------------------------------------------------------------------------
# YAML settings source
# ---------------------------------------------------------------------------
class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load a settings layer from a YAML file.

    A missing file is not an error: ``config/local.yaml`` is optional by design,
    so its absence yields an empty layer. A file that exists but is malformed
    *is* an error, because silently ignoring it would hide the operator's intent.
    """

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self.path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:  # noqa: ARG002
        """Unused: this source supplies the whole mapping at once."""
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        """Return the parsed YAML mapping, or an empty layer if absent."""
        if not self.path.is_file():
            return {}
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(
                f"Could not parse YAML configuration: {self.path}",
                details={"path": str(self.path), "reason": str(exc)},
            ) from exc
        except OSError as exc:
            raise ConfigError(
                f"Could not read configuration file: {self.path}",
                details={"path": str(self.path), "reason": str(exc)},
            ) from exc

        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ConfigError(
                f"Configuration file must contain a mapping at the top level: {self.path}",
                details={"path": str(self.path), "found_type": type(raw).__name__},
            )
        return raw

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r})"


class _Section(BaseModel):
    """Base for configuration sections: strict, coercing, and typo-intolerant."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
class PathsSettings(_Section):
    """Filesystem layout. Relative paths resolve against the project root."""

    base_dir: Path = Field(default_factory=_detect_project_root)
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")

    @model_validator(mode="after")
    def _resolve(self) -> PathsSettings:
        base = self.base_dir.expanduser().resolve()
        object.__setattr__(self, "base_dir", base)
        for name in ("data_dir", "models_dir", "logs_dir"):
            value = Path(getattr(self, name)).expanduser()
            if not value.is_absolute():
                value = base / value
            object.__setattr__(self, name, value)
        return self

    def ensure_directories(self) -> None:
        """Create the writable directories. Called at startup, never at import."""
        for path in (self.data_dir, self.models_dir, self.logs_dir):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigError(
                    f"Cannot create required directory: {path}",
                    details={"path": str(path), "reason": str(exc)},
                ) from exc


class LoggingSettings(_Section):
    """Structured logging configuration."""

    level: LogLevel = "INFO"
    console_format: Literal["pretty", "json"] = "pretty"
    console_colors: bool = True
    file_enabled: bool = True
    file_name: str = "mcmot.jsonl"
    file_max_bytes: int = Field(default=10 * 1024 * 1024, ge=4096)
    file_backup_count: int = Field(default=5, ge=0, le=100)
    metrics_interval_s: float = Field(default=1.0, gt=0.0)
    silence: dict[str, LogLevel] = Field(default_factory=dict)


class DeviceSettings(_Section):
    """Compute device selection and fallback policy."""

    prefer: Literal["auto", "cuda", "cpu"] = "auto"
    index: int = Field(default=0, ge=0)
    allow_fp16: bool = True
    allow_cpu_fallback: bool = True
    min_free_vram_mb: int = Field(default=1024, ge=0)
    warmup_iterations: int = Field(default=3, ge=0, le=50)


class DetectorSettings(_Section):
    """Person detector configuration."""

    name: str = "yolo11"
    weights: Path = Path("yolo11n.pt")
    conf_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    image_size: int = Field(default=640, ge=64, le=2048)
    classes: list[int] = Field(default_factory=lambda: [0])
    max_detections: int = Field(default=100, ge=1, le=1000)
    half: bool = True

    @field_validator("image_size")
    @classmethod
    def _stride_aligned(cls, value: int) -> int:
        if value % 32 != 0:
            raise ValueError("image_size must be a multiple of 32 (YOLO stride)")
        return value

    @field_validator("classes")
    @classmethod
    def _non_empty(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("classes must not be empty")
        return value


class TrackerSettings(_Section):
    """Multi-object tracker configuration (ByteTrack defaults)."""

    name: str = "bytetrack"
    track_high_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    track_low_thresh: float = Field(default=0.1, ge=0.0, le=1.0)
    new_track_thresh: float = Field(default=0.6, ge=0.0, le=1.0)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1, le=1000)
    min_box_area: float = Field(default=100.0, ge=0.0)
    frame_rate: int = Field(default=25, ge=1, le=240)

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> TrackerSettings:
        if self.track_low_thresh > self.track_high_thresh:
            raise ValueError("track_low_thresh must not exceed track_high_thresh")
        return self


class QualitySettings(_Section):
    """Crop quality gate applied before any embedding is computed."""

    min_width: int = Field(default=48, ge=1)
    min_height: int = Field(default=96, ge=1)
    #: Real yolo11n person boxes measure 0.22-0.38; a 0.25 floor rejected legitimate
    #: crops, so shape is a loose sanity check and blur/truncation do the real work.
    min_aspect_ratio: float = Field(default=0.18, gt=0.0)
    max_aspect_ratio: float = Field(default=0.85, gt=0.0)
    blur_threshold: float = Field(default=45.0, ge=0.0)
    max_truncation: float = Field(default=0.15, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _aspect_ordered(self) -> QualitySettings:
        if self.min_aspect_ratio >= self.max_aspect_ratio:
            raise ValueError("min_aspect_ratio must be less than max_aspect_ratio")
        return self


class ReIDSettings(_Section):
    """Person re-identification embedder configuration."""

    name: str = "osnet_ain_x1_0"
    weights: Path = Path("osnet_ain_x1_0.pth")
    backend: Literal["torch", "onnx"] = "torch"
    embedding_dim: int = Field(default=512, ge=32, le=4096)
    input_height: int = Field(default=256, ge=32)
    input_width: int = Field(default=128, ge=32)
    batch_size: int = Field(default=16, ge=1, le=256)
    half: bool = True
    quality: QualitySettings = Field(default_factory=QualitySettings)


class RegistrationSettings(_Section):
    """Enrolment frame-selection policy."""

    target_frames: int = Field(default=30, ge=1, le=200)
    min_accepted_frames: int = Field(default=8, ge=1)
    max_accepted_frames: int = Field(default=20, ge=1)
    duplicate_similarity: float = Field(default=0.97, ge=0.0, le=1.0)
    representative_images: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def _bounds_ordered(self) -> RegistrationSettings:
        if self.min_accepted_frames > self.max_accepted_frames:
            raise ValueError("min_accepted_frames must not exceed max_accepted_frames")
        return self


class IdentitySettings(_Section):
    """Identity resolution thresholds and temporal voting policy.

    ``tau_margin`` is the mechanism a single threshold cannot replace: it
    requires the best candidate to beat the runner-up by a clear gap, which is
    what blocks *confidently wrong* matches between similar-looking people.
    """

    tau_accept: float = Field(default=0.55, ge=-1.0, le=1.0)
    tau_margin: float = Field(default=0.06, ge=0.0, le=2.0)
    top_k: int = Field(default=10, ge=1, le=1000)
    per_person_top: int = Field(default=3, ge=1, le=100)
    vote_window: int = Field(default=15, ge=1, le=1000)
    vote_min_agree: int = Field(default=5, ge=1)
    vote_min_reject: int = Field(default=8, ge=1)
    reid_interval_frames: int = Field(default=10, ge=1, le=1000)
    locked_reid_interval_frames: int = Field(default=60, ge=1, le=10_000)
    max_embeddings_per_person: int = Field(default=20, ge=1, le=500)
    registration: RegistrationSettings = Field(default_factory=RegistrationSettings)

    @model_validator(mode="after")
    def _consistent(self) -> IdentitySettings:
        if self.vote_min_agree > self.vote_window:
            raise ValueError("vote_min_agree must not exceed vote_window")
        if self.per_person_top > self.top_k:
            raise ValueError("per_person_top must not exceed top_k")
        if self.locked_reid_interval_frames < self.reid_interval_frames:
            raise ValueError(
                "locked_reid_interval_frames must be >= reid_interval_frames "
                "(a locked track should be sampled less often, not more)"
            )
        return self


class StorageSettings(_Section):
    """Relational, vector, and blob storage locations.

    Path-like values may contain the ``{data_dir}`` placeholder, expanded once
    against :attr:`PathsSettings.data_dir` after the whole config is assembled.
    """

    database_url: str = "sqlite:///{data_dir}/mcmot.db"
    sqlite_wal: bool = True
    echo_sql: bool = False
    vector_index_path: str = "{data_dir}/index/gallery.faiss"
    blob_dir: str = "{data_dir}/persons"
    rebuild_index_on_mismatch: bool = True


class CameraSettings(_Section):
    """Capture defaults and reconnection policy shared by every camera source."""

    width: int = Field(default=1280, ge=64, le=7680)
    height: int = Field(default=720, ge=64, le=4320)
    target_fps: int = Field(default=25, ge=1, le=240)
    read_timeout_s: float = Field(default=5.0, gt=0.0)
    open_timeout_s: float = Field(default=10.0, gt=0.0)
    buffer_size: int = Field(default=1, ge=1, le=16)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    reconnect_initial_delay_s: float = Field(default=1.0, gt=0.0)
    reconnect_max_delay_s: float = Field(default=30.0, gt=0.0)
    #: 0 means retry forever with exponential backoff.
    max_reconnect_attempts: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _backoff_ordered(self) -> CameraSettings:
        if self.reconnect_initial_delay_s > self.reconnect_max_delay_s:
            raise ValueError("reconnect_initial_delay_s must not exceed reconnect_max_delay_s")
        return self


class PipelineSettings(_Section):
    """Per-camera pipeline behaviour.

    ``queue_size: 1`` plus ``drop_stale_frames`` implements latest-frame-wins.
    Raising the queue size trades latency for smoothness and is almost always
    the wrong call for live tracking: a deeper queue means the pipeline is
    reasoning about the past.
    """

    queue_size: int = Field(default=1, ge=1, le=64)
    drop_stale_frames: bool = True
    max_stale_age_s: float = Field(default=0.5, gt=0.0)
    jpeg_quality: int = Field(default=75, ge=10, le=100)
    stream_max_width: int = Field(default=960, ge=160, le=3840)
    annotate: bool = True


class ApiSettings(_Section):
    """HTTP server configuration."""

    host: str = "0.0.0.0"  # noqa: S104 - LAN access is the point
    port: int = Field(default=8000, ge=1, le=65535)
    docs_enabled: bool = True
    cors_origins: list[str] = Field(default_factory=list)
    request_timeout_s: float = Field(default=30.0, gt=0.0)


# ---------------------------------------------------------------------------
# Root settings object
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """Fully resolved, validated application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="MCMOT__",
        env_nested_delimiter="__",
        env_file=None,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
        validate_default=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    version: str = __version__

    paths: PathsSettings = Field(default_factory=PathsSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    device: DeviceSettings = Field(default_factory=DeviceSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    reid: ReIDSettings = Field(default_factory=ReIDSettings)
    identity: IdentitySettings = Field(default_factory=IdentitySettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    camera: CameraSettings = Field(default_factory=CameraSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    # -- source wiring ----------------------------------------------------
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Declare the layer precedence. Earlier entries win."""
        config_dir = resolve_config_dir()
        return (
            init_settings,
            env_settings,
            cls._dotenv_source(settings_cls, dotenv_settings),
            YamlSettingsSource(settings_cls, config_dir / "local.yaml"),
            YamlSettingsSource(settings_cls, config_dir / "default.yaml"),
            file_secret_settings,
        )

    @classmethod
    def _dotenv_source(
        cls,
        settings_cls: type[BaseSettings],
        provided: PydanticBaseSettingsSource,
    ) -> PydanticBaseSettingsSource:
        """Resolve ``.env`` to an absolute path.

        pydantic-settings resolves a relative ``env_file`` against the current
        working directory, which makes configuration depend on where the process
        was launched from — a service started by a scheduler or a Windows task
        would silently skip the file. Anchoring it to the project root removes
        that failure mode. An explicit ``_env_file`` passed by a caller (tests,
        scripts) always wins.
        """
        if getattr(provided, "env_file", None):
            return provided
        env_file = os.getenv("MCMOT_ENV_FILE") or str(_detect_project_root() / ".env")
        return DotEnvSettingsSource(
            settings_cls,
            env_file=env_file,
            env_file_encoding="utf-8",
            case_sensitive=False,
            env_prefix="MCMOT__",
            env_nested_delimiter="__",
        )

    # -- derived values ---------------------------------------------------
    @model_validator(mode="after")
    def _expand_placeholders(self) -> Settings:
        """Expand ``{data_dir}`` in storage locations and anchor model weights.

        Done here rather than at each use site so that every consumer receives an
        absolute, ready-to-use path and no module has to remember the convention.
        """
        data_dir = self.paths.data_dir.as_posix()
        for name in ("database_url", "vector_index_path", "blob_dir"):
            raw = getattr(self.storage, name)
            try:
                expanded = raw.format(data_dir=data_dir)
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"storage.{name} contains an unknown placeholder: {raw!r} "
                    "(only {data_dir} is supported)"
                ) from exc
            object.__setattr__(self.storage, name, expanded)

        # Model weights are named, not pathed, in config; anchor them to models_dir.
        for section in (self.detector, self.reid):
            weights = Path(section.weights)
            if not weights.is_absolute():
                object.__setattr__(section, "weights", self.paths.models_dir / weights)
        return self

    # -- helpers ----------------------------------------------------------
    @property
    def log_file_path(self) -> Path:
        """Absolute path of the JSON log file."""
        return self.paths.logs_dir / self.logging.file_name

    @property
    def vector_index_file(self) -> Path:
        """Absolute path of the persisted vector index."""
        return Path(self.storage.vector_index_path)

    @property
    def blob_directory(self) -> Path:
        """Absolute path of the person image store."""
        return Path(self.storage.blob_dir)

    def summary(self) -> dict[str, Any]:
        """Return a compact, log- and API-safe view of the active configuration.

        Only operationally interesting values, so that the boot log stays
        readable and ``/api/system/health`` does not become a config dump.
        """
        return {
            "env": self.env,
            "version": self.version,
            "device": {
                "prefer": self.device.prefer,
                "allow_fp16": self.device.allow_fp16,
                "allow_cpu_fallback": self.device.allow_cpu_fallback,
            },
            "detector": {
                "name": self.detector.name,
                "weights": self.detector.weights.name,
                "conf_threshold": self.detector.conf_threshold,
                "image_size": self.detector.image_size,
            },
            "tracker": {"name": self.tracker.name, "match_thresh": self.tracker.match_thresh},
            "reid": {
                "name": self.reid.name,
                "backend": self.reid.backend,
                "embedding_dim": self.reid.embedding_dim,
            },
            "identity": {
                "tau_accept": self.identity.tau_accept,
                "tau_margin": self.identity.tau_margin,
                "top_k": self.identity.top_k,
                "vote_window": self.identity.vote_window,
            },
            "storage": {"database": _redact_url(self.storage.database_url)},
            "pipeline": {
                "queue_size": self.pipeline.queue_size,
                "jpeg_quality": self.pipeline.jpeg_quality,
            },
        }


def _redact_url(url: str) -> str:
    """Strip credentials from a database URL before it reaches a log or a client."""
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}"


def load_settings(
    *,
    config_dir: Path | str | None = None,
    env_file: Path | str | None = None,
    **overrides: Any,
) -> Settings:
    """Build a :class:`Settings` instance, translating validation errors.

    Args:
        config_dir: Directory containing ``default.yaml`` / ``local.yaml``.
            Defaults to ``MCMOT_CONFIG_DIR`` or ``<project root>/config``.
        env_file: Optional dotenv file to layer in. Pass an explicit path in
            tests so a developer's real ``.env`` cannot influence results.
        **overrides: Highest-precedence values, used by tests and scripts.

    Returns:
        A validated settings object.

    Raises:
        ConfigError: If any layer is malformed or any value fails validation.
    """
    kwargs: dict[str, Any] = dict(overrides)
    if env_file is not None:
        kwargs["_env_file"] = str(env_file)

    def _build() -> Settings:
        try:
            return Settings(**kwargs)
        except ConfigError:
            raise  # already precise; do not wrap twice
        except Exception as exc:  # pydantic ValidationError and friends
            raise ConfigError(
                "Configuration validation failed.",
                details={"reason": str(exc)},
            ) from exc

    if config_dir is None:
        return _build()
    with _config_dir_context(Path(config_dir).expanduser().resolve()):
        return _build()
