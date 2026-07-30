"""API tests for the cameras endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestOpen:
    def test_opens_a_camera(self, client: TestClient, video_file: str) -> None:
        response = client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["camera_id"] == "cam-1"
        assert payload["kind"] == "file"

    def test_returns_before_the_stream_is_proven(self, client: TestClient) -> None:
        """A phone that has not launched its camera app yet must not fail the request.

        It should appear as connecting and reconnect on its own, which is what makes
        the Cameras page usable while someone walks across the room with a handset.
        """
        response = client.post(
            "/api/cameras", json={"camera_id": "phone", "source": "rtsp://192.0.2.1:554/live"}
        )

        assert response.status_code == 201
        assert response.json()["state"] in {"idle", "connecting", "reconnecting", "failed"}

    def test_rejects_a_duplicate_id(self, client: TestClient, video_file: str) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        response = client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        assert response.status_code == 409

    def test_rejects_an_unrecognisable_source(self, client: TestClient) -> None:
        response = client.post(
            "/api/cameras", json={"camera_id": "cam-1", "source": "not-a-camera"}
        )

        assert response.status_code == 422
        assert response.json()["code"] == "camera.invalid_source"

    def test_rejects_a_path_unsafe_id(self, client: TestClient, video_file: str) -> None:
        """The id lands in URLs and in log context, so it is constrained at the edge."""
        response = client.post(
            "/api/cameras", json={"camera_id": "../../etc/passwd", "source": video_file}
        )

        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid"

    def test_leaves_no_entry_behind_after_a_failed_open(self, client: TestClient) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": "not-a-camera"})

        assert client.get("/api/cameras").json()["cameras"] == []


class TestList:
    def test_empty(self, client: TestClient) -> None:
        assert client.get("/api/cameras").json() == {"cameras": []}

    def test_reports_capture_statistics(self, client: TestClient, video_file: str) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        camera = client.get("/api/cameras").json()["cameras"][0]

        assert camera["frames_delivered"] >= 0
        assert camera["frames_dropped"] >= 0
        assert camera["reconnects"] == 0


class TestGet:
    def test_returns_one_camera(self, client: TestClient, video_file: str) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        response = client.get("/api/cameras/cam-1")

        assert response.status_code == 200
        assert response.json()["camera_id"] == "cam-1"

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        response = client.get("/api/cameras/nope")

        assert response.status_code == 404
        assert response.json()["code"] == "camera.not_found"

    def test_credentials_are_redacted(self, client: TestClient) -> None:
        """An RTSP password must not come back out of the API it went into."""
        client.post(
            "/api/cameras",
            json={"camera_id": "ip", "source": "rtsp://admin:s3cret@192.0.2.9:554/stream"},
        )

        source = client.get("/api/cameras/ip").json()["source"]

        assert "s3cret" not in source


class TestClose:
    def test_closes_a_camera(self, client: TestClient, video_file: str) -> None:
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})

        response = client.delete("/api/cameras/cam-1")

        assert response.status_code == 204
        assert client.get("/api/cameras/cam-1").status_code == 404

    def test_stops_a_running_pipeline_first(self, client: TestClient, video_file: str) -> None:
        """Closing the source under a running worker would leave it spinning."""
        client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})
        client.post("/api/tracking/cam-1/start")

        client.delete("/api/cameras/cam-1")

        assert client.get("/api/tracking").json()["pipelines"] == []

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        assert client.delete("/api/cameras/nope").status_code == 404
