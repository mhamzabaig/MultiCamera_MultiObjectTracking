"""Domain contracts shared across every module.

Pure data: no I/O, no framework imports, no dependency on any other MC-MOT
package. Modules depend on these types instead of on each other, which is what
makes the detector, tracker, ReID model, and storage backend genuinely swappable
rather than nominally so.
"""

from __future__ import annotations

from mcmot.domain.detection import PERSON_CLASS_ID, Detection
from mcmot.domain.frame import Frame, FrameMeta
from mcmot.domain.geometry import BBox
from mcmot.domain.identity import (
    UNKNOWN_LABEL,
    IdentityCandidate,
    IdentityStatus,
    IdentityVerdict,
)
from mcmot.domain.person import (
    EmbeddingRecord,
    PersonRecord,
    new_person_id,
    utc_now,
)
from mcmot.domain.track import Track, TrackState
from mcmot.domain.types import (
    EmbeddingArray,
    EmbeddingMatrix,
    ImageArray,
    ScoreArray,
)

__all__ = [
    "PERSON_CLASS_ID",
    "UNKNOWN_LABEL",
    "BBox",
    "Detection",
    "EmbeddingArray",
    "EmbeddingMatrix",
    "EmbeddingRecord",
    "Frame",
    "FrameMeta",
    "ImageArray",
    "ScoreArray",
    "IdentityCandidate",
    "IdentityStatus",
    "IdentityVerdict",
    "PersonRecord",
    "Track",
    "TrackState",
    "new_person_id",
    "utc_now",
]
