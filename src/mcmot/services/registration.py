"""Registration: uploaded frames in, an enrolled person out.

**The server crops, not the browser.** A registration frame is a whole photo of a
room, and the gallery needs a tight person crop. Cropping here rather than in
JavaScript means one implementation of "what is a person", shared with the tracking
path, so enrolment and inference agree — and it means a buggy or hostile client
cannot poison the gallery with whatever rectangle it felt like sending.

**The largest detection wins.** Registration is a deliberate act: the subject is
holding the camera or standing in front of it, so they are the nearest and largest
person in frame. A bystander further back is smaller. Taking the largest is a simple
rule that matches how people actually register, and it keeps someone walking past in
the background out of the gallery.

Frames with nobody in them are skipped rather than fatal. A capture burst of thirty
browser frames will contain a few where the subject turned away or the exposure
blew out, and failing the whole registration for those would be maddening. Too few
usable frames is still an error — raised by the identity service, which owns that
threshold.
"""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np

from mcmot.core.errors import IdentityError, InvalidImageError
from mcmot.core.logging import get_logger
from mcmot.detection.base import BaseDetector
from mcmot.domain.person import PersonRecord
from mcmot.domain.types import ImageArray
from mcmot.identity.service import IdentityService

__all__ = ["RegistrationService"]

_log = get_logger(__name__)


class RegistrationService:
    """Turns uploaded image bytes into an enrolled identity."""

    def __init__(self, *, detector: BaseDetector, identity: IdentityService) -> None:
        """Wire the service.

        Args:
            detector: Person detector, shared with the tracking pipeline.
            identity: Identity service that owns enrolment.
        """
        self._detector = detector
        self._identity = identity

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
        crops: list[ImageArray] = []
        for payload in images:
            image = self._decode(payload)
            if image is None:
                continue
            decoded += 1
            crop = self._largest_person(image)
            if crop is not None:
                crops.append(crop)

        if decoded == 0:
            raise InvalidImageError(
                "None of the uploaded frames could be decoded as an image.",
                details={"supplied": len(images)},
            )
        if not crops:
            raise IdentityError(
                "No person was detected in any of the uploaded frames. "
                "Make sure the whole person is visible and well lit.",
                details={"supplied": len(images), "decoded": decoded},
            )

        _log.info(
            "registration.frames_prepared",
            name=name,
            supplied=len(images),
            decoded=decoded,
            with_person=len(crops),
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

    def _largest_person(self, image: ImageArray) -> ImageArray | None:
        """Return a crop of the largest detected person, or ``None`` if there is none."""
        detections = self._detector.detect(image)
        if not detections:
            return None
        best = max(detections, key=lambda detection: detection.bbox.area)
        height, width = image.shape[0], image.shape[1]
        x1, y1, x2, y2 = best.bbox.clip(width, height).to_int()
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:  # pragma: no cover - clip() already guarantees a valid box
            return None
        # Copied, not viewed: the crop is a window into a frame that is about to go
        # out of scope, and the gallery keeps it well past that point.
        return np.ascontiguousarray(crop, dtype=np.uint8)
