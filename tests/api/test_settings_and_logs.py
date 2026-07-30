"""API tests for the settings and logs endpoints."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient


class TestSettings:
    def test_returns_the_active_configuration(self, client: TestClient) -> None:
        response = client.get("/api/system/settings")

        assert response.status_code == 200
        payload = response.json()
        assert payload["config"]["identity"]["tau_accept"] == 0.55
        assert payload["editable"] is False

    def test_matches_what_health_reports(self, client: TestClient) -> None:
        """One summary, two endpoints: the page and the boot log cannot disagree."""
        settings = client.get("/api/system/settings").json()["config"]
        health = client.get("/api/system/health").json()["config"]

        assert settings == health

    def test_reports_which_files_were_loaded(self, client: TestClient) -> None:
        """Why a value is what it is cannot be answered without knowing the layers."""
        payload = client.get("/api/system/settings").json()

        assert isinstance(payload["config_files"], list)
        assert all(name.endswith(".yaml") for name in payload["config_files"])

    def test_is_not_writable(self, client: TestClient) -> None:
        """Thresholds are validated once at boot; there is deliberately no setter."""
        for method in (client.put, client.post, client.patch):
            response = method("/api/system/settings", json={"identity": {"tau_accept": 0.9}})
            assert response.status_code == 405


class TestLogs:
    def test_returns_recent_entries(self, client: TestClient) -> None:
        client.get("/api/system/version")

        response = client.get("/api/system/logs")

        assert response.status_code == 200
        payload = response.json()
        assert payload["file"]
        events = [entry.get("event") for entry in payload["entries"]]
        assert "app.started" in events

    def test_respects_the_limit(self, client: TestClient) -> None:
        for _ in range(10):
            client.get("/api/system/version")

        payload = client.get("/api/system/logs", params={"limit": 3}).json()

        assert len(payload["entries"]) == 3
        assert payload["truncated"] is True

    def test_returns_the_newest_entries_last(self, client: TestClient) -> None:
        """Read top to bottom like a terminal, so the newest line is at the bottom."""
        client.get("/api/system/version")
        entries = client.get("/api/system/logs").json()["entries"]

        timestamps = [entry["timestamp"] for entry in entries if entry.get("timestamp")]

        assert timestamps == sorted(timestamps)

    def test_filters_by_level(self, client: TestClient) -> None:
        payload = client.get("/api/system/logs", params={"level": "error"}).json()

        assert all(entry["level"] == "error" for entry in payload["entries"])

    def test_rejects_an_unknown_level(self, client: TestClient) -> None:
        response = client.get("/api/system/logs", params={"level": "chatty"})

        assert response.status_code == 422

    def test_survives_a_half_written_line(self, client: TestClient, app: Any) -> None:
        """A log being appended to while read will sometimes end mid-record."""
        client.get("/api/system/version")
        path = app.state.container.settings.log_file_path
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"event": "truncated", "lev\n')

        response = client.get("/api/system/logs")

        assert response.status_code == 200
        assert response.json()["entries"]
