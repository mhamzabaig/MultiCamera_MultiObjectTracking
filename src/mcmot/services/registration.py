"""Registration: uploaded frames in, an enrolled person out.

**The server crops, not the browser.** A registration frame is a whole photo of a
room, and the gallery needs a tight person crop. Cropping here rather than in
JavaScript means one implementation of "what is a person", shared with the tracking
path, so enrolment and inference agree — and it means a buggy or hostile client
cannot poison the gallery with whatever rectangle it felt like sending.

**Picking the subject when more than one person is in frame.** Registration is a
deliberate act: the subject stands in front of the camera, so they are the nearest,
sharpest, most completely visible person there. "Largest box" alone is not enough —
a bystander closer to the lens, or one standing at the frame edge half out of shot,
can win on area while being a far worse subject. So candidates are scored on how much
usable person they actually show:

* **quality** — the crop quality gate's own score: pixel area, sharpness, and how
  much of the box is cut off by the frame edge. This is "visibility and features"
  made numeric, and it is the same gate that judges tracking crops.
* **confidence** — the detector's own certainty, which falls for partial and
  occluded people.
* **centrality** — someone registering faces the camera; someone walking past does
  not stop in the middle of the frame.

**An ambiguous frame is skipped, not guessed.** If the best candidate is not clearly
ahead of the runner-up, the frame contributes nothing. Guessing costs far more than
skipping: a gallery that mixes two people will then recognise *both* of them under
the one name, which is the worst failure this system has. There are thirty frames in
a burst and only eight are needed, so skipping is nearly free.

Frames with nobody in them are skipped rather than fatal, for the same reason: a
capture burst will contain a few where the subject turned away or the exposure blew
out. Too few usable frames is still an error — raised by the identity service, which
owns that threshold and which independently checks that the frames that *did* survive
all show the same person.
"""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np

from mcmot.core.errors import IdentityError, InvalidImageError
from mcmot.core.logging import get_logger
from mcmot.detection.base import BaseDetector
from mcmot.domain.detection import Detection
from mcmot.domain.person import PersonRecord
from mcmot.domain.types import ImageArray
from mcmot.identity.service import IdentityService
from mcmot.reid.quality import QualityGate

__all__ = ["RegistrationService"]

_log = get_logger(__name__)

#: How far ahead of the runner-up the best candidate must score to be treated as the
#: subject. Below this the frame is skipped: with thirty frames in a burst and eight
#: needed, skipping an ambiguous one is nearly free, while guessing wrong merges two
#: people into a single identity.
_SUBJECT_MARGIN = 0.06

#: Distance from frame centre to a corner in normalised coordinates, used to scale
#: centrality onto [0, 1].
_MAX_OFFSET = math.hypot(0.5, 0.5)


class RegistrationService:
    """Turns uploaded image bytes into an enrolled identity."""

    def __init__(
        self, *, detector: BaseDetector, identity: IdentityService, quality: QualityGate
    ) -> None:
        """Wire the service.

        Args:
            detector: Person detector, shared with the tracking pipeline.
            identity: Identity service that owns enrolment.
            quality: Crop quality gate, shared with the tracking pipeline so
                registration and inference agree on what a usable crop is.
        """
        self._detector = detector
        self._identity = identity
        self._quality = quality

    def register(self, name: str, images: list[bytes]) -> PersonRecord:
        """Enrol a person from uploaded frames.

        Args:
            name: Display name; must be unique.
            images: Encoded frames (JPEG or PNG) in capture order.

        Returns:
            The created person record.

        Raises:
            InvalidImageError: If no frame could be decoded at all.
            IdentityError: If no frame contained a detectable person, or if too few
                survive the quality gate.
            PersonAlreadyExistsError: If the name is taken.
        """
        if not images:
            raise IdentityError("No frames were uploaded.")

        decoded = 0
        ambiguous = 0
        crops: list[ImageArray] = []
        for payload in images:
            image = self._decode(payload)
            if image is None:
                continue
            decoded += 1
            crop, reason = self._subject(image)
            if crop is not None:
                crops.append(crop)
            elif reason == "ambiguous":
                ambiguous += 1

        if decoded == 0:
            raise InvalidImageError(
                "None of the uploaded frames could be decoded as an image.",
                details={"supplied": len(images)},
            )
        if not crops:
            message = (
                "No person was detected in any of the uploaded frames. "
                "Make sure the whole person is visible and well lit."
            )
            if ambiguous:
                # A different problem with a different remedy, so say so rather than
                # claiming nobody was found.
                message = (
                    "Could not tell which person to register: every frame showed "
                    "more than one person and none of them stood out. Register with "
                    "only the subject in view."
                )
            raise IdentityError(
                message,
                details={"supplied": len(images), "decoded": decoded, "ambiguous": ambiguous},
            )

        _log.info(
            "registration.frames_prepared",
            name=name,
            supplied=len(images),
            decoded=decoded,
            with_person=len(crops),
            ambiguous=ambiguous,
        )
        return self._identity.enrol(name, crops)

    # -- internals --------------------------------------------------------
    @staticmethod
    def _decode(payload: bytes) -> ImageArray | None:
        """Decode one upload, or ``None`` if it is not a usable image.

        Returns ``None`` rather than raising: one corrupt frame in a burst of thirty
        is a normal browser artefact, and the caller decides when too few is fatal.
        """
        if not payload:
            return None
        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return None
        # IMREAD_COLOR guarantees 3-channel BGR uint8, which is what the rest of the
        # pipeline's contracts require; OpenCV's stubs are simply looser than that.
        return cast("ImageArray", image)

    def _subject(self, image: ImageArray) -> tuple[ImageArray | None, str | None]:
        """Return a crop of the person being registered.

        Returns:
            ``(crop, None)`` when one person is clearly the subject, or
            ``(None, reason)`` where reason is ``"no_person"`` or ``"ambiguous"``.
        """
        detections = self._detector.detect(image)
        if not detections:
            return None, "no_person"

        height, width = image.shape[0], image.shape[1]
        scored: list[tuple[float, ImageArray]] = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox.clip(width, height).to_int()
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:  # pragma: no cover - clip() guarantees a valid box
                continue
            # Copied, not viewed: the crop is a window into a frame about to go out
            # of scope, and the gallery keeps it well past that point.
            crop = np.ascontiguousarray(crop, dtype=np.uint8)
            scored.append((self._subject_score(detection, crop, (height, width)), crop))

        if not scored:  # pragma: no cover - every detection yields some crop
            return None, "no_person"

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_crop = scored[0]
        if best_score <= 0.0:
            # Every candidate failed the quality gate: too small, too blurry, or cut
            # off by the frame edge.
            return None, "no_person"
        if len(scored) > 1 and best_score - scored[1][0] < _SUBJECT_MARGIN:
            return None, "ambiguous"
        return best_crop, None

    def _subject_score(
        self, detection: Detection, crop: ImageArray, frame_shape: tuple[int, int]
    ) -> float:
        """Rank how likely a detection is to be the person deliberately registering.

        Zero means unusable — a crop the quality gate rejected outright, which no
        weighting should be able to rescue.
        """
        verdict = self._quality.assess(crop, bbox=detection.bbox, frame_shape=frame_shape)
        if not verdict.accepted:
            return 0.0

        height, width = frame_shape
        centre_x, centre_y = detection.bbox.center
        # 1.0 dead centre, falling to 0.0 at a corner. Someone registering faces the
        # camera; someone walking past does not stop in the middle of the frame.
        offset = math.hypot((centre_x / width) - 0.5, (centre_y / height) - 0.5)
        centrality = max(0.0, 1.0 - offset / _MAX_OFFSET)

        # Quality dominates because it already carries size, sharpness, and how much
        # of the person is cut off — the three things that decide how many usable
        # features a crop contains. Confidence and centrality break ties between
        # candidates the gate is equally happy with.
        return 0.6 * verdict.score + 0.2 * detection.score + 0.2 * centrality
