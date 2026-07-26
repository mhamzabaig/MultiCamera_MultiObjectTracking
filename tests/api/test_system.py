"""API tests for the system endpoints and the shared error contract.

Every failing response in this system has the same shape. Asserting that here,
once, is what lets phases 9 and 10 add endpoints and a frontend without
re-litigating the error format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcmot.api.middleware import REQUEST_ID_HEADER
from mcmot.container import Container
from mcmot.core.errors import CameraDisconnectedError, PersonNotFoundError
from mcmot.core.settings import load_settings
from mcmot.main import create_app


def read_log_events(path: Path) -> list[dict]:
    """Parse our JSON log file into events.

    Filters by parseability rather than by content: third-party libraries (httpx
    inside TestClient, for instance) also route through the same handler, and a
    substring search over the raw text would match their lines too.
    """
    events: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:  # pragma: no cover
            continue
    return events


@pytest.fixture(autouse=True)
def reset_logging():
    yield
    structlog.reset_defaults()
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()


@pytest.fixture
def app(config_dir: Path, tmp_path: Path) -> FastAPI:
    """An app confined to tmp_path and pinned to CPU."""
    settings = load_settings(
        config_dir=config_dir,
        env="test",
        paths={"base_dir": str(tmp_path / "workspace")},
        device={"prefer": "cpu"},
    )
    return create_app(settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """A client whose context manager runs the real lifespan (startup/shutdown)."""
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_returns_ok(self, client: TestClient) -> None:
        response = client.get("/api/system/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_payload_matches_the_declared_schema(self, client: TestClient) -> None:
        payload = client.get("/api/system/health").json()

        assert payload["env"] == "test"
        assert payload["version"]
        assert payload["uptime_s"] >= 0.0
        assert payload["started_epoch"] is not None
        assert payload["device"]["type"] == "cpu"
        assert payload["device"]["fp16"] is False
        assert payload["config"]["identity"]["tau_accept"] == 0.55

    def test_no_unexpected_top_level_fields(self, client: TestClient) -> None:
        """extra="forbid" on the response model turns drift into a test failure."""
        assert set(client.get("/api/system/health").json()) == {
            "status",
            "version",
            "env",
            "uptime_s",
            "started_epoch",
            "device",
            "config",
        }

    def test_configuration_summary_is_redacted(self, client: TestClient) -> None:
        config = client.get("/api/system/health").json()["config"]

        assert "weights" in config["detector"]
        assert "database" in config["storage"]


class TestVersion:
    def test_returns_build_identity(self, client: TestClient) -> None:
        response = client.get("/api/system/version")

        assert response.status_code == 200
        assert response.json() == {
            "name": "mcmot",
            "version": client.get("/api/system/health").json()["version"],
            "env": "test",
        }


class TestRootAndDocs:
    def test_root_describes_the_service(self, client: TestClient) -> None:
        payload = client.get("/").json()

        assert payload["name"] == "mcmot"
        assert payload["health"] == "/api/system/health"

    def test_openapi_schema_is_served(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()

        assert "/api/system/health" in schema["paths"]

    def test_docs_can_be_disabled(self, config_dir: Path, tmp_path: Path) -> None:
        settings = load_settings(
            config_dir=config_dir,
            env="test",
            paths={"base_dir": str(tmp_path / "workspace")},
            device={"prefer": "cpu"},
            api={"docs_enabled": False},
        )
        with TestClient(create_app(settings)) as client:
            assert client.get("/docs").status_code == 404
            assert client.get("/openapi.json").status_code == 404


class TestRequestCorrelation:
    def test_response_carries_a_request_id(self, client: TestClient) -> None:
        response = client.get("/api/system/version")

        assert response.headers[REQUEST_ID_HEADER]

    def test_ids_differ_between_requests(self, client: TestClient) -> None:
        first = client.get("/api/system/version").headers[REQUEST_ID_HEADER]
        second = client.get("/api/system/version").headers[REQUEST_ID_HEADER]

        assert first != second

    def test_inbound_request_id_is_preserved(self, client: TestClient) -> None:
        """A correlation started by a proxy or the frontend must survive the hop."""
        response = client.get("/api/system/version", headers={REQUEST_ID_HEADER: "trace-abc-123"})

        assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"


class TestErrorContract:
    def test_unknown_route_uses_the_problem_shape(self, client: TestClient) -> None:
        response = client.get("/api/system/does-not-exist")

        assert response.status_code == 404
        payload = response.json()
        assert payload["code"] == "http.not_found"
        assert payload["status"] == 404
        assert payload["request_id"]

    def test_domain_errors_map_to_their_declared_status(
        self, app: FastAPI, client: TestClient
    ) -> None:
        @app.get("/test/boom")
        async def boom() -> None:
            raise CameraDisconnectedError(
                "cam-1 dropped", details={"camera_id": "cam-1", "attempts": 3}
            )

        response = client.get("/test/boom")

        assert response.status_code == 502
        payload = response.json()
        assert payload["code"] == "camera.disconnected"
        assert payload["message"] == "cam-1 dropped"
        assert payload["details"] == {"camera_id": "cam-1", "attempts": 3}

    def test_not_found_domain_errors_map_to_404(self, app: FastAPI, client: TestClient) -> None:
        @app.get("/test/missing-person")
        async def missing() -> None:
            raise PersonNotFoundError(details={"person_id": "abc"})

        response = client.get("/test/missing-person")

        assert response.status_code == 404
        assert response.json()["code"] == "person.not_found"

    def test_unexpected_exceptions_do_not_leak_internals(self, app: FastAPI) -> None:
        """A traceback in a browser response is an information-disclosure bug.

        ``raise_server_exceptions=False`` makes TestClient behave like a real
        server: Starlette re-raises unhandled exceptions after building the
        response so the process can log them, and only a real ASGI server
        swallows that. With the default of ``True`` we would be asserting against
        the test harness rather than against what a browser receives.
        """

        @app.get("/test/bug")
        async def bug() -> None:
            raise RuntimeError("secret internal detail")

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/test/bug")

        assert response.status_code == 500
        payload = response.json()
        assert payload["code"] == "internal.error"
        assert payload["message"] == "An unexpected internal error occurred."
        assert "secret internal detail" not in response.text
        assert "RuntimeError" not in response.text
        assert payload["request_id"]

    def test_validation_errors_report_the_offending_field(
        self, app: FastAPI, client: TestClient
    ) -> None:
        @app.get("/test/typed")
        async def typed(count: int) -> dict[str, int]:
            return {"count": count}

        response = client.get("/test/typed", params={"count": "not-a-number"})

        assert response.status_code == 422
        payload = response.json()
        assert payload["code"] == "request.invalid"
        assert payload["details"]["errors"][0]["location"] == ["query", "count"]

    def test_error_bodies_use_the_problem_media_type(self, client: TestClient) -> None:
        response = client.get("/api/system/nope")

        assert response.headers["content-type"].startswith("application/problem+json")


class TestLifespan:
    def test_startup_and_shutdown_run(self, app: FastAPI) -> None:
        container: Container = app.state.container
        assert container.started_epoch is None

        with TestClient(app) as client:
            client.get("/api/system/health")
            # `started_epoch`, not `uptime_s > 0`: time.monotonic() has ~15 ms
            # granularity on Windows, and this whole block runs in under one.
            assert container.started_epoch is not None
            assert container.uptime_s >= 0.0

        events = [event["event"] for event in read_log_events(container.settings.log_file_path)]
        assert "app.started" in events
        assert "app.stopped" in events

    def test_health_polling_is_excluded_from_access_logs(
        self, app: FastAPI, client: TestClient
    ) -> None:
        """A polled health check would otherwise dominate the log stream."""
        container: Container = app.state.container
        client.get("/api/system/health")
        client.get("/api/system/version")

        logged_paths = [
            event["path"]
            for event in read_log_events(container.settings.log_file_path)
            if event.get("event") == "http.request"
        ]

        assert "/api/system/version" in logged_paths
        assert "/api/system/health" not in logged_paths
