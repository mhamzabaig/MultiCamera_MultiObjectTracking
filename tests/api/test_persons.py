"""API tests for the persons endpoints."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

Frames = Any  # the make_frames fixture; typed loosely to avoid a cross-module import


class TestRegister:
    def test_enrols_a_person_from_uploaded_frames(
        self, client: TestClient, make_frames: Frames
    ) -> None:
        response = client.post("/api/persons", data={"name": "Hamza"}, files=make_frames(hue=60))

        assert response.status_code == 201, response.text
        payload = response.json()
        assert payload["name"] == "Hamza"
        assert payload["id"]
        assert payload["embedding_count"] >= 1

    def test_stores_more_than_one_embedding(self, client: TestClient, make_frames: Frames) -> None:
        """A single-vector gallery is the fragile case multi-shot enrolment avoids."""
        payload = client.post(
            "/api/persons", data={"name": "Hamza"}, files=make_frames(hue=60, count=20)
        ).json()

        assert payload["embedding_count"] > 1

    def test_indexes_what_it_stores(self, client: TestClient, registered: dict[str, Any]) -> None:
        """The database and the derived index must agree straight after enrolment."""
        stats = client.get("/api/persons").json()["stats"]

        assert stats["persons"] == 1
        assert stats["embeddings"] == registered["embedding_count"]
        assert stats["index_vectors"] == stats["embeddings"]

    def test_rejects_a_duplicate_name(
        self, client: TestClient, registered: dict[str, Any], make_frames: Frames
    ) -> None:
        response = client.post("/api/persons", data={"name": "Hamza"}, files=make_frames(hue=60))

        assert response.status_code == 409
        assert response.json()["code"] == "person.already_exists"

    def test_rejects_frames_with_nobody_in_them(
        self, client: TestClient, fake_detector: Any, make_frames: Frames
    ) -> None:
        """Enrolling an empty room would poison the gallery with background."""
        fake_detector.find_nobody = True

        response = client.post("/api/persons", data={"name": "Ghost"}, files=make_frames(hue=60))

        assert response.status_code == 422
        payload = response.json()
        assert payload["code"] == "identity.error"
        assert "No person was detected" in payload["message"]

    def test_rejects_undecodable_uploads(self, client: TestClient) -> None:
        files = [("frames", ("junk.jpg", b"not an image at all", "image/jpeg"))]

        response = client.post("/api/persons", data={"name": "Junk"}, files=files)

        assert response.status_code == 422
        assert response.json()["code"] == "image.invalid"

    def test_rejects_too_few_usable_frames(self, client: TestClient, make_frames: Frames) -> None:
        """Below the minimum the gallery cannot cover pose variety, so it is refused."""
        response = client.post(
            "/api/persons", data={"name": "Brief"}, files=make_frames(hue=60, count=2)
        )

        assert response.status_code == 422
        payload = response.json()
        assert payload["code"] == "identity.error"
        assert payload["details"]["required"] >= 3

    def test_requires_a_name(self, client: TestClient, make_frames: Frames) -> None:
        response = client.post("/api/persons", files=make_frames(hue=60))

        assert response.status_code == 422
        assert response.json()["code"] == "request.invalid"

    def test_nothing_is_persisted_when_registration_fails(
        self, client: TestClient, fake_detector: Any, make_frames: Frames
    ) -> None:
        """A failed enrolment must not leave a person with an empty gallery."""
        fake_detector.find_nobody = True
        client.post("/api/persons", data={"name": "Ghost"}, files=make_frames(hue=60))

        assert client.get("/api/persons").json()["persons"] == []


class TestList:
    def test_empty_gallery(self, client: TestClient) -> None:
        payload = client.get("/api/persons").json()

        assert payload["persons"] == []
        assert payload["stats"]["persons"] == 0

    def test_lists_several_people(self, client: TestClient, make_frames: Frames) -> None:
        client.post("/api/persons", data={"name": "Hamza"}, files=make_frames(hue=60))
        client.post("/api/persons", data={"name": "Ali"}, files=make_frames(hue=180))

        names = {person["name"] for person in client.get("/api/persons").json()["persons"]}

        assert names == {"Hamza", "Ali"}


class TestGet:
    def test_returns_one_person(self, client: TestClient, registered: dict[str, Any]) -> None:
        response = client.get(f"/api/persons/{registered['id']}")

        assert response.status_code == 200
        assert response.json()["name"] == "Hamza"

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        response = client.get("/api/persons/does-not-exist")

        assert response.status_code == 404
        assert response.json()["code"] == "person.not_found"


class TestImage:
    def test_serves_the_representative_image(
        self, client: TestClient, registered: dict[str, Any]
    ) -> None:
        response = client.get(f"/api/persons/{registered['id']}/image")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert len(response.content) > 0

    def test_unknown_person_is_404(self, client: TestClient) -> None:
        assert client.get("/api/persons/nope/image").status_code == 404


class TestRename:
    def test_changes_the_name(self, client: TestClient, registered: dict[str, Any]) -> None:
        response = client.patch(f"/api/persons/{registered['id']}", json={"name": "Hamza Baig"})

        assert response.status_code == 200
        assert response.json()["name"] == "Hamza Baig"

    def test_keeps_the_gallery(self, client: TestClient, registered: dict[str, Any]) -> None:
        """Renaming relabels an identity; it must not re-enrol it."""
        client.patch(f"/api/persons/{registered['id']}", json={"name": "Hamza Baig"})

        payload = client.get(f"/api/persons/{registered['id']}").json()

        assert payload["embedding_count"] == registered["embedding_count"]

    def test_rejects_a_taken_name(
        self, client: TestClient, registered: dict[str, Any], make_frames: Frames
    ) -> None:
        client.post("/api/persons", data={"name": "Ali"}, files=make_frames(hue=180))

        response = client.patch(f"/api/persons/{registered['id']}", json={"name": "Ali"})

        assert response.status_code == 409

    def test_rejects_a_blank_name(self, client: TestClient, registered: dict[str, Any]) -> None:
        response = client.patch(f"/api/persons/{registered['id']}", json={"name": ""})

        assert response.status_code == 422

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        response = client.patch("/api/persons/nope", json={"name": "Anyone"})

        assert response.status_code == 404


class TestDelete:
    def test_removes_the_person(self, client: TestClient, registered: dict[str, Any]) -> None:
        response = client.delete(f"/api/persons/{registered['id']}")

        assert response.status_code == 204
        assert client.get(f"/api/persons/{registered['id']}").status_code == 404

    def test_removes_the_vectors_too(self, client: TestClient, registered: dict[str, Any]) -> None:
        """A vector left behind would keep matching a person who was deleted."""
        client.delete(f"/api/persons/{registered['id']}")

        stats = client.get("/api/persons").json()["stats"]

        assert stats["embeddings"] == 0
        assert stats["index_vectors"] == 0

    def test_unknown_id_is_404(self, client: TestClient) -> None:
        assert client.delete("/api/persons/nope").status_code == 404


class TestRebuildIndex:
    def test_reports_the_vector_count(self, client: TestClient, registered: dict[str, Any]) -> None:
        response = client.post("/api/persons/index/rebuild")

        assert response.status_code == 200
        assert response.json()["vectors"] == registered["embedding_count"]

    def test_heals_a_divergent_index(self, client: TestClient, registered: dict[str, Any]) -> None:
        """The index is a derived cache; the database is the source of truth."""
        container = client.app.state.container  # type: ignore[attr-defined]
        container.vector_index.reset()
        assert client.get("/api/persons").json()["stats"]["index_vectors"] == 0

        client.post("/api/persons/index/rebuild")

        stats = client.get("/api/persons").json()["stats"]
        assert stats["index_vectors"] == stats["embeddings"]
