"""API tests for the tracking endpoints, including the MJPEG stream."""

from __future__ import annotations

import time
from typing import Any

import pytest
from fastapi.testclient import TestClient


def wait_for(predicate: Any, *, timeout: float = 10.0) -> bool:
    """Poll until ``predicate()`` is true. Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def tracking(client: TestClient, video_file: str) -> TestClient:
    """A client with one camera open and its pipeline running."""
    assert (
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file}).status_code
        == 201
    )
    assert client.post("/api/tracking/cam-1/start").status_code == 200
    return client


class TestStart:
    def test_starts_a_pipeline(self, client: TestClient, video_file: str) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        response = client.post("/api/tracking/cam-1/start")

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["camera_id"] == "cam-1"
        assert payload["running"] is True

    def test_processes_frames(self, tracking: TestClient) -> None:
        # Two frames, not one: a rate cannot be measured from a single event, and the
        # worker correctly reports 0 until it has an interval to divide by.
        assert wait_for(
            lambda: tracking.get("/api/tracking/cam-1/status").json()["frames_processed"] > 1
        )

        payload = tracking.get("/api/tracking/cam-1/status").json()
        assert payload["people"] >= 1  # the fake detector always finds one
        assert payload["fps"] > 0

    def test_rejects_a_camera_that_is_not_open(self, client: TestClient) -> None:
        response = client.post("/api/tracking/nope/start")

        assert response.status_code == 404
        assert response.json()["code"] == "camera.not_found"

    def test_rejects_a_second_start(self, tracking: TestClient) -> None:
        """Two workers on one camera would double GPU cost and fight over track ids."""
        response = tracking.post("/api/tracking/cam-1/start")

        assert response.status_code == 500
        assert response.json()["code"] == "pipeline.error"


class TestList:
    def test_empty(self, client: TestClient) -> None:
        assert client.get("/api/tracking").json() == {"pipelines": []}

    def test_lists_the_running_pipeline(self, tracking: TestClient) -> None:
        pipelines = tracking.get("/api/tracking").json()["pipelines"]

        assert [pipeline["camera_id"] for pipeline in pipelines] == ["cam-1"]


class TestStatus:
    def test_unknown_camera_is_404(self, client: TestClient) -> None:
        response = client.get("/api/tracking/nope/status")

        assert response.status_code == 404

    def test_identified_never_exceeds_people(self, tracking: TestClient) -> None:
        wait_for(lambda: tracking.get("/api/tracking/cam-1/status").json()["frames_processed"] > 3)

        payload = tracking.get("/api/tracking/cam-1/status").json()

        assert payload["identified"] <= payload["people"]


class TestStop:
    def test_stops_the_pipeline(self, tracking: TestClient) -> None:
        response = tracking.post("/api/tracking/cam-1/stop")

        assert response.status_code == 204
        assert tracking.get("/api/tracking").json()["pipelines"] == []

    def test_leaves_the_camera_open(self, tracking: TestClient) -> None:
        """Stopping tracking must not force a reconnect to inspect the camera."""
        tracking.post("/api/tracking/cam-1/stop")

        assert tracking.get("/api/cameras/cam-1").status_code == 200

    def test_unknown_camera_is_404(self, client: TestClient) -> None:
        assert client.post("/api/tracking/nope/stop").status_code == 404

    def test_can_restart_after_stopping(self, tracking: TestClient) -> None:
        tracking.post("/api/tracking/cam-1/stop")

        response = tracking.post("/api/tracking/cam-1/start")

        assert response.status_code == 200
        assert response.json()["running"] is True


class TestStream:
    def test_serves_multipart_mjpeg(self, tracking: TestClient) -> None:
        assert wait_for(
            lambda: tracking.get("/api/tracking/cam-1/status").json()["frames_processed"] > 0
        )

        with tracking.stream("GET", "/api/tracking/cam-1/stream") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("multipart/x-mixed-replace")
            assert response.headers["cache-control"].startswith("no-store")

            chunks = []
            for chunk in response.iter_bytes():
                chunks.append(chunk)
                if len(chunks) >= 2:
                    break

        payload = b"".join(chunks)
        assert b"Content-Type: image/jpeg" in payload
        # JPEG start-of-image marker: the part really is an image, not a placeholder.
        assert b"\xff\xd8" in payload

    def test_unknown_camera_is_404_rather_than_an_empty_stream(self, client: TestClient) -> None:
        """Resolving the source before responding is what makes this a clean error."""
        response = client.get("/api/tracking/nope/stream")

        assert response.status_code == 404
        assert response.json()["code"] == "camera.not_found"

    def test_a_camera_without_tracking_is_404(self, client: TestClient, video_file: str) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        assert client.get("/api/tracking/cam-1/stream").status_code == 404
