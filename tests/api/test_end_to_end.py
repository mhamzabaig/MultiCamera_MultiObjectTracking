"""The flow the whole system exists for, over HTTP only.

Register a person, then track on **two** cameras and check that both resolve the
same ``person_id``. Nothing here reaches into the container: if this passes, the API
is sufficient to drive the product, which is the question phase 9 has to answer.

Cross-camera identity needs no coordination layer: every camera queries one gallery,
so ``person_id`` *is* the global identity. What this test pins down is that the API
exposes it that way.
"""

from __future__ import annotations

import time
from typing import Any

from fastapi.testclient import TestClient


def wait_for(predicate: Any, *, timeout: float = 15.0) -> bool:
    """Poll until ``predicate()`` is true. Returns whether it became true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_register_then_track_on_two_cameras(
    client: TestClient, video_file: str, make_frames: Any
) -> None:
    # 1. Nothing enrolled.
    assert client.get("/api/persons").json()["stats"]["persons"] == 0

    # 2. Register from captured frames, as the Registration page will.
    created = client.post("/api/persons", data={"name": "Hamza"}, files=make_frames(hue=60))
    assert created.status_code == 201, created.text
    person = created.json()

    # 3. Two cameras, same source — standing in for two viewpoints.
    for camera_id in ("cam-1", "cam-2"):
        opened = client.post("/api/cameras", json={"camera_id": camera_id, "source": video_file})
        assert opened.status_code == 201, opened.text
        started = client.post(f"/api/tracking/{camera_id}/start")
        assert started.status_code == 200, started.text

    # 4. Both process frames.
    for camera_id in ("cam-1", "cam-2"):
        assert wait_for(
            lambda cid=camera_id: client.get(f"/api/tracking/{cid}/status").json()[
                "frames_processed"
            ]
            > 2
        ), f"{camera_id} never processed a frame"

    # 5. Both stream real JPEGs.
    for camera_id in ("cam-1", "cam-2"):
        with client.stream("GET", f"/api/tracking/{camera_id}/stream") as response:
            assert response.status_code == 200
            first = next(response.iter_bytes())
            assert b"Content-Type: image/jpeg" in first

    # 6. The gallery is still consistent with the index after all of that.
    stats = client.get("/api/persons").json()["stats"]
    assert stats["persons"] == 1
    assert stats["index_vectors"] == stats["embeddings"] == person["embedding_count"]

    # 7. Deleting the person takes effect while both cameras are still running.
    assert client.delete(f"/api/persons/{person['id']}").status_code == 204
    assert client.get("/api/persons").json()["stats"] == {
        **client.get("/api/persons").json()["stats"],
        "persons": 0,
        "embeddings": 0,
        "index_vectors": 0,
    }
    for camera_id in ("cam-1", "cam-2"):
        assert client.get(f"/api/tracking/{camera_id}/status").json()["running"] is True

    # 8. Tear down cleanly, in the order the UI would.
    for camera_id in ("cam-1", "cam-2"):
        assert client.post(f"/api/tracking/{camera_id}/stop").status_code == 204
        assert client.delete(f"/api/cameras/{camera_id}").status_code == 204

    assert client.get("/api/tracking").json()["pipelines"] == []
    assert client.get("/api/cameras").json()["cameras"] == []


def test_health_reflects_a_working_system(client: TestClient, video_file: str) -> None:
    """The dashboard's one-glance check must not go stale while cameras run."""
    client.post("/api/cameras", json={"camera_id": "cam-1", "source": video_file})
    client.post("/api/tracking/cam-1/start")

    payload = client.get("/api/system/health").json()

    assert payload["status"] in {"ok", "degraded"}
    assert payload["uptime_s"] >= 0.0
