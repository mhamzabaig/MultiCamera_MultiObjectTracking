"""Fixtures for API tests: a real app wired with fake models.

The container's collaborators are ``cached_property``, which caches into the
instance ``__dict__`` — so seeding that dict *before* anything touches the property
substitutes a fake. That is the same mechanism the container's own shutdown uses to
tell whether a resource was ever constructed.

Everything below the models is real: real SQLite, real FAISS, real quality gate,
real routing, real error handling. Only the two things that need CUDA and 100 MB of
weights are faked, and the crops are generated with genuine texture so they pass the
real quality gate rather than a relaxed one.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from mcmot.container import Container
from mcmot.core.settings import load_settings
from mcmot.domain.detection import Detection
from mcmot.domain.geometry import BBox
from mcmot.domain.types import EmbeddingMatrix, ImageArray
from mcmot.main import create_app

EMBEDDING_DIM = 512


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeDetector:
    """Finds one person filling most of the frame, in a plausible person shape."""

    def __init__(self) -> None:
        self.calls = 0
        self.find_nobody = False

    def detect(self, image: ImageArray) -> list[Detection]:
        self.calls += 1
        if self.find_nobody:
            return []
        height, width = image.shape[0], image.shape[1]
        # Inset so the box is a person-ish 0.4 aspect and never frame-truncated.
        margin_x = width * 0.3
        return [
            Detection(
                bbox=BBox(margin_x, height * 0.05, width - margin_x, height * 0.95),
                score=0.92,
            )
        ]

    def warmup(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-detector", "weights": "none", "device": "cpu"}


class FakeEmbedder:
    """Deterministic embeddings derived from a crop's own pixels.

    Content-derived rather than random, so two crops of the same synthetic "person"
    match each other and two different ones do not — which is what makes an
    end-to-end registration-then-recognition assertion meaningful without a real
    ReID model.
    """

    dim = EMBEDDING_DIM

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, crops: list[ImageArray]) -> EmbeddingMatrix:
        self.calls += 1
        vectors = np.zeros((len(crops), self.dim), dtype=np.float32)
        for row, crop in enumerate(crops):
            # Mean colour drives the direction; the rest is a fixed pattern, so the
            # vector is stable for a given colour and differs between colours.
            mean = crop.reshape(-1, 3).mean(axis=0) / 255.0
            base = np.linspace(0.1, 1.0, self.dim, dtype=np.float32)
            vectors[row] = base * (1.0 + mean[0] * 4.0) + mean[1] * 2.0 + mean[2]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def extract_one(self, crop: ImageArray) -> Any:
        return self.extract([crop])[0]

    def warmup(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": "fake-embedder", "dim": self.dim}


# ---------------------------------------------------------------------------
# Synthetic imagery
# ---------------------------------------------------------------------------
def person_frame(*, seed: int = 0, hue: int = 60, size: tuple[int, int] = (480, 320)) -> bytes:
    """Encode a textured synthetic frame, JPEG.

    Textured on purpose: a flat colour has zero Laplacian variance and the real
    quality gate would reject it as blurred, which would mean testing the failure
    path by accident.
    """
    import cv2

    height, width = size
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 60, size=(height, width, 3), dtype=np.uint8)
    # A distinct dominant colour so the fake embedder can tell people apart.
    image[:, :, 0] = np.clip(image[:, :, 0].astype(np.int16) + hue, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return bytes(buffer.tobytes())


def frames_for(hue: int, count: int = 12) -> list[tuple[str, tuple[str, io.BytesIO, str]]]:
    """Build a multipart payload of ``count`` frames of one synthetic person."""
    return [
        ("frames", (f"frame-{i}.jpg", io.BytesIO(person_frame(seed=i, hue=hue)), "image/jpeg"))
        for i in range(count)
    ]


@pytest.fixture
def make_frames() -> Callable[..., list[tuple[str, tuple[str, io.BytesIO, str]]]]:
    """Expose :func:`frames_for` as a fixture.

    A fixture rather than a cross-module import: ``tests/`` is deliberately not a
    package, so importing between test modules would depend on pytest's sys.path
    insertion instead of on something declared.
    """
    return frames_for


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
@pytest.fixture
def container(config_dir: Path, tmp_path: Path) -> Container:
    """A CPU container whose detector and embedder are fakes."""
    settings = load_settings(
        config_dir=config_dir,
        env="test",
        paths={"base_dir": str(tmp_path / "workspace")},
        device={"prefer": "cpu"},
        reid={"embedding_dim": EMBEDDING_DIM},
    )
    instance = Container(settings)
    instance.__dict__["detector"] = FakeDetector()
    instance.__dict__["embedder"] = FakeEmbedder()
    return instance


@pytest.fixture
def fake_detector(container: Container) -> FakeDetector:
    detector: FakeDetector = container.__dict__["detector"]
    return detector


@pytest.fixture
def app(container: Container) -> FastAPI:
    return create_app(container=container)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def video_file(tmp_path: Path) -> str:
    """A short generated clip, standing in for a camera.

    A file rather than device index 0: a test suite must not depend on a webcam
    existing, being free, or having anyone in front of it. Phase 7 paces non-live
    sources to their own frame rate, so a file behaves like a 25 FPS camera rather
    than being consumed in a burst.
    """
    import cv2

    path = tmp_path / "camera.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 480))
    rng = np.random.default_rng(7)
    for _ in range(200):  # 8 seconds, so a test never races the end of the clip
        writer.write(rng.integers(0, 200, size=(480, 320, 3), dtype=np.uint8))
    writer.release()
    assert path.exists()
    return str(path)


@pytest.fixture
def registered(client: TestClient) -> dict[str, Any]:
    """One enrolled person, for tests that need an existing identity."""
    response = client.post("/api/persons", data={"name": "Hamza"}, files=frames_for(hue=60))
    assert response.status_code == 201, response.text
    payload: dict[str, Any] = response.json()
    return payload
