"""Structured logging.

Two sinks, one event stream:

* **Console** — human-readable, coloured, for the terminal you are watching.
* **File** — newline-delimited JSON in ``logs/mcmot.jsonl``, rotated, for
  grepping and post-mortems (``Get-Content mcmot.jsonl | ConvertFrom-Json``).

Both are fed by the same structlog pipeline, so a field bound anywhere appears
in both places. Standard-library loggers (uvicorn, SQLAlchemy, Ultralytics) are
routed through the same formatter, so third-party output does not arrive in a
different shape.

Volume discipline
-----------------
At 25 FPS across three cameras, logging one line per detection is ~75 lines per
second — enough noise to bury the events that matter. Two tools address this:

* :class:`Throttle` for periodic aggregates ("camera 1: 24.8 FPS, 3 people").
* :func:`bind_context` for correlation fields, so a *single* event line carries
  ``camera_id`` / ``track_id`` / ``request_id`` instead of being logged repeatedly.

The rule for this codebase: **log state changes and aggregates, never frames.**
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import structlog

from mcmot.core.errors import ConfigError
from mcmot.core.settings import LoggingSettings

__all__ = [
    "Throttle",
    "bind_context",
    "configure_logging",
    "get_logger",
    "is_configured",
]

_configured = False
_configure_lock = threading.Lock()

#: Processors shared by both sinks and by foreign (stdlib) log records.
_SHARED_PROCESSORS: tuple[Any, ...] = (
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.UnicodeDecoder(),
)


def is_configured() -> bool:
    """Return whether logging has already been initialised."""
    return _configured


def configure_logging(config: LoggingSettings, *, log_dir: Path, force: bool = False) -> None:
    """Initialise the logging stack. Idempotent unless ``force`` is set.

    Args:
        config: Validated logging configuration.
        log_dir: Directory for the rotating JSON file.
        force: Re-configure even if already initialised (used by tests).

    Raises:
        ConfigError: If the log file cannot be opened for writing.
    """
    global _configured
    with _configure_lock:
        if _configured and not force:
            return

        structlog.configure(
            processors=[
                *_SHARED_PROCESSORS,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=False,
        )

        handlers: list[logging.Handler] = [_build_console_handler(config)]
        if config.file_enabled:
            handlers.append(_build_file_handler(config, log_dir))

        root = logging.getLogger()
        for existing in list(root.handlers):
            root.removeHandler(existing)
            # Release file locks; on Windows a lingering handle blocks rotation.
            if isinstance(existing, logging.FileHandler):
                existing.close()
        for handler in handlers:
            root.addHandler(handler)
        root.setLevel(config.level)

        # Third-party loggers that are chatty at our chosen level.
        for name, level in config.silence.items():
            logging.getLogger(name).setLevel(level)

        _configured = True


def _build_console_handler(config: LoggingSettings) -> logging.Handler:
    renderer: Any
    if config.console_format == "pretty":
        renderer = structlog.dev.ConsoleRenderer(colors=config.console_colors)
    else:
        renderer = structlog.processors.JSONRenderer()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=list(_SHARED_PROCESSORS),
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    return handler


def _build_file_handler(config: LoggingSettings, log_dir: Path) -> logging.Handler:
    path = log_dir / config.file_name
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=config.file_max_bytes,
            backupCount=config.file_backup_count,
            encoding="utf-8",
            delay=False,
        )
    except OSError as exc:
        raise ConfigError(
            f"Cannot open log file for writing: {path}",
            details={"path": str(path), "reason": str(exc)},
        ) from exc

    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=list(_SHARED_PROCESSORS),
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                # Structured tracebacks: exceptions stay machine-readable
                # instead of collapsing into one giant escaped string.
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    return handler


def get_logger(name: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a bound logger.

    Args:
        name: Logger name, conventionally the module's ``__name__``.
        **initial_values: Fields bound to every event from this logger, e.g.
            ``get_logger(__name__, camera_id="cam-1")``.
    """
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_values:
        return logger.bind(**initial_values)
    return logger


@contextmanager
def bind_context(**values: Any) -> Iterator[None]:
    """Bind correlation fields for the duration of the block.

    Backed by context variables, so the fields follow the logical flow of work
    across ``await`` boundaries and are isolated per thread and per task — a
    request handler and a camera worker never see each other's context.

    Example:
        >>> with bind_context(camera_id="cam-1"):
        ...     get_logger(__name__).info("stream.opened")
    """
    tokens = structlog.contextvars.bind_contextvars(**values)
    try:
        yield
    finally:
        structlog.contextvars.reset_contextvars(**tokens)


class Throttle:
    """Rate limiter for periodic log lines and metric emissions.

    Keyed, so several cameras throttle independently while sharing one instance.
    Uses a monotonic clock, therefore immune to wall-clock adjustments (NTP
    steps, DST) that would otherwise stall or flood emissions.

    Example:
        >>> throttle = Throttle(1.0)
        >>> if throttle.ready("cam-1"):
        ...     log.info("camera.metrics", fps=24.8, people=3)
    """

    __slots__ = ("_interval", "_last", "_lock")

    def __init__(self, interval_s: float) -> None:
        """Args: interval_s: Minimum seconds between emissions for a given key."""
        if interval_s < 0:
            raise ValueError("interval_s must be non-negative")
        self._interval = interval_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def ready(self, key: str = "") -> bool:
        """Return True at most once per interval per key, and consume the slot."""
        now = time.monotonic()
        with self._lock:
            previous = self._last.get(key)
            if previous is not None and (now - previous) < self._interval:
                return False
            self._last[key] = now
            return True

    def reset(self, key: str | None = None) -> None:
        """Forget throttle state for one key, or for all keys when ``key`` is None."""
        with self._lock:
            if key is None:
                self._last.clear()
            else:
                self._last.pop(key, None)
