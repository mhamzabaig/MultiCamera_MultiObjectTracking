"""Structured logging tests.

The JSON file is a machine interface: dashboards, log shipping, and post-mortem
greps depend on its field names, so the shape is asserted rather than assumed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest
import structlog

from mcmot.core.logging import (
    Throttle,
    bind_context,
    configure_logging,
    get_logger,
)
from mcmot.core.settings import LoggingSettings


@pytest.fixture(autouse=True)
def reset_structlog():
    """Leave global logging state clean for every other test."""
    yield
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def log_dir(tmp_path: Path) -> Path:
    path = tmp_path / "logs"
    path.mkdir()
    return path


def read_events(log_dir: Path, name: str = "mcmot.jsonl") -> list[dict]:
    """Parse the JSON log file into event dictionaries."""
    path = log_dir / name
    logging.getLogger().handlers[-1].flush()
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line]


def configure(log_dir: Path, **overrides) -> None:
    configure_logging(LoggingSettings(**overrides), log_dir=log_dir, force=True)


class TestJsonSink:
    def test_event_is_written_with_the_expected_fields(self, log_dir: Path) -> None:
        configure(log_dir)

        get_logger("test.module").info("camera.opened", camera_id="cam-1", fps=25)

        (event,) = read_events(log_dir)
        assert event["event"] == "camera.opened"
        assert event["camera_id"] == "cam-1"
        assert event["fps"] == 25
        assert event["level"] == "info"
        assert event["logger"] == "test.module"
        assert event["timestamp"].endswith("Z")

    def test_file_sink_can_be_disabled(self, log_dir: Path) -> None:
        configure(log_dir, file_enabled=False)

        get_logger("test").info("nothing.persisted")

        assert not (log_dir / "mcmot.jsonl").exists()

    def test_custom_file_name_is_honoured(self, log_dir: Path) -> None:
        configure(log_dir, file_name="tracking.jsonl")

        get_logger("test").info("event")

        assert (log_dir / "tracking.jsonl").is_file()

    def test_level_filters_lower_severities(self, log_dir: Path) -> None:
        configure(log_dir, level="WARNING")

        log = get_logger("test")
        log.debug("dropped.debug")
        log.info("dropped.info")
        log.warning("kept.warning")

        events = read_events(log_dir)
        assert [event["event"] for event in events] == ["kept.warning"]

    def test_exceptions_are_recorded_structurally(self, log_dir: Path) -> None:
        """dict_tracebacks keeps a traceback queryable instead of one long string."""
        configure(log_dir)

        try:
            raise ValueError("index rebuild failed")
        except ValueError:
            get_logger("test").exception("index.rebuild_failed")

        (event,) = read_events(log_dir)
        assert event["level"] == "error"
        assert "exception" in event
        assert "ValueError" in json.dumps(event["exception"])

    def test_stdlib_loggers_share_the_same_stream(self, log_dir: Path) -> None:
        """Third-party output (uvicorn, ultralytics) must not bypass the format."""
        configure(log_dir)

        logging.getLogger("uvicorn.error").warning("third party message")

        (event,) = read_events(log_dir)
        assert event["event"] == "third party message"
        assert event["logger"] == "uvicorn.error"

    def test_silence_config_raises_noisy_logger_levels(self, log_dir: Path) -> None:
        configure(log_dir, silence={"uvicorn.access": "ERROR"})

        logging.getLogger("uvicorn.access").warning("suppressed")
        logging.getLogger("uvicorn.access").error("kept")

        assert [e["event"] for e in read_events(log_dir)] == ["kept"]


class TestContextBinding:
    def test_bound_fields_appear_on_events(self, log_dir: Path) -> None:
        configure(log_dir)

        with bind_context(camera_id="cam-2", request_id="abc123"):
            get_logger("test").info("tracking.started")

        (event,) = read_events(log_dir)
        assert event["camera_id"] == "cam-2"
        assert event["request_id"] == "abc123"

    def test_context_is_removed_on_exit(self, log_dir: Path) -> None:
        configure(log_dir)

        with bind_context(camera_id="cam-2"):
            pass
        get_logger("test").info("after.context")

        (event,) = read_events(log_dir)
        assert "camera_id" not in event

    def test_context_is_restored_after_nesting(self, log_dir: Path) -> None:
        configure(log_dir)

        with bind_context(camera_id="cam-1"):
            with bind_context(camera_id="cam-2"):
                get_logger("test").info("inner")
            get_logger("test").info("outer")

        inner, outer = read_events(log_dir)
        assert inner["camera_id"] == "cam-2"
        assert outer["camera_id"] == "cam-1"

    def test_context_is_cleared_even_when_the_block_raises(self, log_dir: Path) -> None:
        configure(log_dir)

        with pytest.raises(RuntimeError), bind_context(camera_id="cam-9"):
            raise RuntimeError("boom")
        get_logger("test").info("after.failure")

        (event,) = read_events(log_dir)
        assert "camera_id" not in event

    def test_initial_values_bind_to_every_event(self, log_dir: Path) -> None:
        configure(log_dir)
        log = get_logger("test", camera_id="cam-7")

        log.info("first")
        log.info("second")

        assert [e["camera_id"] for e in read_events(log_dir)] == ["cam-7", "cam-7"]


class TestIdempotency:
    def test_repeat_configuration_is_ignored_without_force(self, log_dir: Path) -> None:
        configure(log_dir)
        handler_count = len(logging.getLogger().handlers)

        configure_logging(LoggingSettings(), log_dir=log_dir, force=False)

        assert len(logging.getLogger().handlers) == handler_count

    def test_force_replaces_handlers_rather_than_stacking(self, log_dir: Path) -> None:
        """Duplicated handlers would double every line in the file."""
        configure(log_dir)
        configure(log_dir)
        configure(log_dir)

        get_logger("test").info("once")

        assert len(read_events(log_dir)) == 1


class TestThrottle:
    def test_first_call_is_allowed(self) -> None:
        assert Throttle(10.0).ready("cam-1") is True

    def test_subsequent_calls_within_the_interval_are_suppressed(self) -> None:
        throttle = Throttle(10.0)

        assert throttle.ready("cam-1") is True
        assert throttle.ready("cam-1") is False
        assert throttle.ready("cam-1") is False

    def test_keys_are_throttled_independently(self) -> None:
        """Three cameras must not silence each other's metrics."""
        throttle = Throttle(10.0)

        assert throttle.ready("cam-1") is True
        assert throttle.ready("cam-2") is True
        assert throttle.ready("cam-3") is True
        assert throttle.ready("cam-1") is False

    def test_slot_reopens_after_the_interval(self) -> None:
        throttle = Throttle(0.02)

        assert throttle.ready() is True
        time.sleep(0.03)
        assert throttle.ready() is True

    def test_zero_interval_never_suppresses(self) -> None:
        throttle = Throttle(0.0)

        assert throttle.ready() is True
        assert throttle.ready() is True

    def test_reset_clears_one_key(self) -> None:
        throttle = Throttle(10.0)
        throttle.ready("cam-1")
        throttle.ready("cam-2")

        throttle.reset("cam-1")

        assert throttle.ready("cam-1") is True
        assert throttle.ready("cam-2") is False

    def test_reset_clears_every_key(self) -> None:
        throttle = Throttle(10.0)
        throttle.ready("cam-1")
        throttle.ready("cam-2")

        throttle.reset()

        assert throttle.ready("cam-1") is True
        assert throttle.ready("cam-2") is True

    def test_negative_interval_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Throttle(-1.0)
