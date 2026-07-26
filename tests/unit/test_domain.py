"""Domain contract tests.

These types are the interfaces every module speaks. The geometry in particular is
worth pinning: an off-by-one in cropping or an inverted aspect ratio would show up
much later as unexplained ReID accuracy loss.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcmot.domain import (
    BBox,
    Detection,
    EmbeddingRecord,
    Frame,
    FrameMeta,
    IdentityCandidate,
    IdentityStatus,
    IdentityVerdict,
    PersonRecord,
    Track,
    TrackState,
    new_person_id,
    utc_now,
)


class TestBBox:
    def test_measurements(self) -> None:
        box = BBox(10, 20, 50, 120)

        assert box.width == 40
        assert box.height == 100
        assert box.area == 4000
        assert box.center == (30.0, 70.0)
        assert box.aspect_ratio == pytest.approx(0.4)

    def test_inverted_box_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid bounding box"):
            BBox(50, 20, 10, 120)

    def test_degenerate_box_is_allowed(self) -> None:
        """Zero-area boxes occur legitimately as clipped or predicted boxes."""
        box = BBox(10, 10, 10, 10)

        assert box.area == 0
        assert box.aspect_ratio == float("inf")

    def test_is_immutable(self) -> None:
        box = BBox(0, 0, 10, 10)

        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            box.x1 = 5  # type: ignore[misc]

    @pytest.mark.parametrize(
        ("converter", "expected"),
        [
            ("to_xyxy", (10.0, 20.0, 50.0, 120.0)),
            ("to_xywh", (10.0, 20.0, 40.0, 100.0)),
            ("to_cxcywh", (30.0, 70.0, 40.0, 100.0)),
        ],
    )
    def test_conversions(self, converter: str, expected: tuple[float, ...]) -> None:
        assert getattr(BBox(10, 20, 50, 120), converter)() == expected

    def test_round_trips_through_every_representation(self) -> None:
        box = BBox(10.5, 20.25, 50.5, 120.25)

        assert BBox.from_xywh(*box.to_xywh()) == box
        assert BBox.from_cxcywh(*box.to_cxcywh()) == box

    def test_to_int_rounds_outward(self) -> None:
        """Rounding inward would silently drop a boundary pixel from every crop."""
        assert BBox(10.6, 20.2, 50.1, 120.9).to_int() == (10, 20, 51, 121)

    def test_clip_confines_the_box_to_the_frame(self) -> None:
        clipped = BBox(-30, -10, 700, 500).clip(640, 480)

        assert clipped.to_xyxy() == (0.0, 0.0, 640.0, 480.0)

    def test_clip_of_a_fully_contained_box_is_identity(self) -> None:
        box = BBox(10, 10, 100, 200)

        assert box.clip(640, 480) == box

    @pytest.mark.parametrize(
        ("box", "expected"),
        [
            (BBox(10, 10, 110, 210), 0.0),
            (BBox(-50, 0, 50, 100), pytest.approx(0.5)),
            (BBox(700, 500, 800, 600), 1.0),
        ],
        ids=["inside", "half-out", "fully-out"],
    )
    def test_truncation_measures_the_fraction_outside(self, box: BBox, expected) -> None:
        assert box.truncation(640, 480) == expected

    def test_expand_grows_about_the_centre(self) -> None:
        expanded = BBox(100, 100, 200, 300).expand(0.1)

        assert expanded.center == (150.0, 200.0)
        assert expanded.width == pytest.approx(110.0)
        assert expanded.height == pytest.approx(220.0)

    def test_expand_accepts_shrinking(self) -> None:
        assert BBox(0, 0, 100, 100).expand(-0.5).width == pytest.approx(50.0)

    def test_expand_rejects_full_collapse(self) -> None:
        with pytest.raises(ValueError, match="greater than -1.0"):
            BBox(0, 0, 10, 10).expand(-1.0)

    @pytest.mark.parametrize(
        ("other", "expected"),
        [
            (BBox(0, 0, 100, 100), 1.0),
            (BBox(200, 200, 300, 300), 0.0),
            (BBox(100, 0, 200, 100), 0.0),
            (BBox(50, 0, 150, 100), pytest.approx(1 / 3)),
        ],
        ids=["identical", "disjoint", "edge-touching", "half-overlap"],
    )
    def test_iou(self, other: BBox, expected) -> None:
        assert BBox(0, 0, 100, 100).iou(other) == expected

    def test_iou_is_symmetric(self) -> None:
        a, b = BBox(0, 0, 100, 100), BBox(40, 40, 160, 160)

        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_iou_with_a_degenerate_box_is_zero(self) -> None:
        assert BBox(0, 0, 100, 100).iou(BBox(50, 50, 50, 50)) == 0.0


class TestDetection:
    def test_defaults_to_the_person_class(self) -> None:
        detection = Detection(BBox(0, 0, 10, 20), score=0.9)

        assert detection.class_id == 0
        assert detection.is_person

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_out_of_range_score_is_rejected(self, score: float) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            Detection(BBox(0, 0, 10, 20), score=score)

    def test_serialisation(self) -> None:
        payload = Detection(BBox(1, 2, 3, 4), score=0.987654).to_dict()

        assert payload == {"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9877, "class_id": 0}


class TestTrack:
    def test_defaults(self) -> None:
        track = Track(track_id=7, bbox=BBox(0, 0, 10, 20), score=0.8)

        assert track.state is TrackState.CONFIRMED
        assert track.is_active

    @pytest.mark.parametrize(
        ("state", "time_since_update", "active"),
        [
            (TrackState.CONFIRMED, 0, True),
            (TrackState.CONFIRMED, 3, False),
            (TrackState.TENTATIVE, 0, False),
            (TrackState.LOST, 0, False),
        ],
    )
    def test_is_active_requires_a_current_association(
        self, state: TrackState, time_since_update: int, active: bool
    ) -> None:
        track = Track(
            track_id=1,
            bbox=BBox(0, 0, 10, 20),
            score=0.8,
            state=state,
            time_since_update=time_since_update,
        )

        assert track.is_active is active

    def test_state_serialises_as_a_plain_string(self) -> None:
        assert TrackState.LOST.value == "lost"
        assert Track(1, BBox(0, 0, 1, 1), 0.5, state=TrackState.LOST).to_dict()["state"] == "lost"


class TestIdentityVerdict:
    def test_identified_verdict_displays_the_name(self) -> None:
        verdict = IdentityVerdict(
            status=IdentityStatus.IDENTIFIED, person_id="p1", name="Hamza", score=0.91
        )

        assert verdict.display_name == "Hamza"
        assert verdict.is_identified

    def test_unknown_verdict_never_leaks_a_candidate_name(self) -> None:
        """A rejected candidate's name must not reach the frame overlay."""
        verdict = IdentityVerdict.unknown(
            score=0.41,
            candidates=(IdentityCandidate(person_id="p1", name="Hamza", score=0.41),),
        )

        assert verdict.display_name == "Unknown"
        assert verdict.is_identified is False
        assert verdict.to_dict()["name"] == "Unknown"

    def test_rejection_context_is_preserved_for_calibration(self) -> None:
        verdict = IdentityVerdict.unknown(score=0.52, margin=0.01)

        assert verdict.score == 0.52
        assert verdict.margin == 0.01

    def test_pending_is_distinct_from_unknown(self) -> None:
        assert IdentityVerdict.pending().status is IdentityStatus.PENDING
        assert IdentityVerdict.unknown().status is IdentityStatus.UNKNOWN

    def test_runner_up_is_exposed(self) -> None:
        verdict = IdentityVerdict(
            status=IdentityStatus.IDENTIFIED,
            person_id="p1",
            name="Hamza",
            score=0.9,
            candidates=(
                IdentityCandidate("p1", "Hamza", 0.90),
                IdentityCandidate("p2", "Ali", 0.62),
            ),
        )

        assert verdict.runner_up is not None
        assert verdict.runner_up.name == "Ali"

    def test_runner_up_is_none_with_a_single_candidate(self) -> None:
        verdict = IdentityVerdict(
            status=IdentityStatus.IDENTIFIED,
            candidates=(IdentityCandidate("p1", "Hamza", 0.9),),
        )

        assert verdict.runner_up is None

    def test_candidates_are_excluded_from_the_default_payload(self) -> None:
        verdict = IdentityVerdict(
            status=IdentityStatus.IDENTIFIED,
            name="Hamza",
            candidates=(IdentityCandidate("p1", "Hamza", 0.9),),
        )

        assert "candidates" not in verdict.to_dict()
        assert len(verdict.to_dict(include_candidates=True)["candidates"]) == 1


class TestFrame:
    @staticmethod
    def image(height: int = 48, width: int = 64) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)

    def test_create_derives_dimensions(self) -> None:
        frame = Frame.create(self.image(480, 640), camera_id="cam-1", index=12)

        assert frame.meta.width == 640
        assert frame.meta.height == 480
        assert frame.shape == (480, 640)
        assert frame.meta.camera_id == "cam-1"
        assert frame.meta.index == 12

    def test_grayscale_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="HxWx3"):
            Frame(image=np.zeros((48, 64), dtype=np.uint8), meta=self.meta())

    def test_float_input_is_rejected(self) -> None:
        """Normalised float arrays reaching the pipeline are a bug, not a variant."""
        with pytest.raises(ValueError, match="uint8"):
            Frame(image=np.zeros((48, 64, 3), dtype=np.float32), meta=self.meta())

    def test_copy_image_returns_an_independent_writable_array(self) -> None:
        frame = Frame.create(self.image(), camera_id="cam-1", index=0)

        copy = frame.copy_image()
        copy[0, 0, 0] = 255

        assert frame.image[0, 0, 0] == 0

    def test_age_is_measured_on_the_monotonic_clock(self) -> None:
        meta = self.meta()

        assert meta.age_s(meta.monotonic + 0.25) == pytest.approx(0.25)

    @staticmethod
    def meta() -> FrameMeta:
        return FrameMeta(camera_id="cam-1", index=0, timestamp=0.0, width=64, height=48)


class TestPersonRecord:
    def test_generated_ids_are_unique_and_opaque(self) -> None:
        first, second = new_person_id(), new_person_id()

        assert first != second
        assert len(first) == 32
        assert first.isalnum()

    def test_timestamps_are_timezone_aware(self) -> None:
        """Naive datetimes in SQLite become unattributable timestamps."""
        assert utc_now().tzinfo is not None
        assert PersonRecord(name="Hamza").created_at.tzinfo is not None

    def test_serialisation_uses_iso_timestamps(self) -> None:
        payload = PersonRecord(name="Hamza", embedding_count=12).to_dict()

        assert payload["name"] == "Hamza"
        assert payload["embedding_count"] == 12
        assert "T" in payload["created_at"]

    def test_metadata_is_isolated_between_instances(self) -> None:
        first, second = PersonRecord(name="a"), PersonRecord(name="b")

        first.metadata["role"] = "staff"

        assert second.metadata == {}


class TestEmbeddingRecord:
    def test_valid_embedding(self) -> None:
        vector = np.zeros(512, dtype=np.float32)
        record = EmbeddingRecord(vector_id=1, person_id="p1", vector=vector)

        assert record.dim == 512

    def test_two_dimensional_vector_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            EmbeddingRecord(
                vector_id=1, person_id="p1", vector=np.zeros((1, 512), dtype=np.float32)
            )

    def test_float64_vector_is_rejected(self) -> None:
        """float64 would double index memory and break FAISS's expected dtype."""
        with pytest.raises(ValueError, match="float32"):
            EmbeddingRecord(vector_id=1, person_id="p1", vector=np.zeros(512))
