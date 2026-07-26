"""Identity Service output.

The vocabulary of recognition. Three ideas are encoded deliberately:

**"Unknown" is a first-class answer, not a failure.** Below threshold, the system
reports :attr:`IdentityStatus.UNKNOWN` rather than guessing. Putting a wrong name
on a person is worse than admitting ignorance, so the type system makes the
distinction unavoidable at every call site.

**Runner-up scores travel with the verdict.** :attr:`IdentityVerdict.margin`
carries the gap to the second-best *person*, which is what separates a confident
match from a coin flip between two similar-looking people. A single similarity
score cannot express that.

**Identity is a property of a track, not a frame.** :attr:`IdentityVerdict.locked`
and :attr:`vote_count` expose the temporal state that keeps a displayed label
stable instead of flickering between candidates every frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = ["IdentityCandidate", "IdentityStatus", "IdentityVerdict"]

UNKNOWN_LABEL = "Unknown"


class IdentityStatus(str, Enum):
    """Outcome of an identity resolution."""

    IDENTIFIED = "identified"
    """Passed both the similarity threshold and the margin test."""

    UNKNOWN = "unknown"
    """Searched, but no candidate met the acceptance criteria."""

    PENDING = "pending"
    """Not yet resolved: too few votes, or no usable crop has passed the quality
    gate. Distinct from UNKNOWN so the UI can show "identifying…" instead of
    prematurely labelling someone a stranger."""

    DISABLED = "disabled"
    """Recognition is switched off, or the gallery is empty. Reported explicitly
    so an empty database never looks like a system that identifies nobody."""


@dataclass(frozen=True, slots=True)
class IdentityCandidate:
    """One person's aggregated similarity for a single query."""

    person_id: str
    name: str
    score: float
    """Aggregated cosine similarity in ``[-1, 1]``: the mean of this person's best
    ``identity.per_person_top`` embedding hits."""

    matched_embeddings: int = 1
    """How many of this person's gallery vectors appeared in the top-K."""

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view."""
        return {
            "person_id": self.person_id,
            "name": self.name,
            "score": round(self.score, 4),
            "matched_embeddings": self.matched_embeddings,
        }


@dataclass(frozen=True, slots=True)
class IdentityVerdict:
    """The Identity Service's answer for one track at one point in time."""

    status: IdentityStatus
    person_id: str | None = None
    name: str | None = None
    score: float = 0.0
    """Similarity of the winning candidate, or of the best rejected one."""

    margin: float = 0.0
    """``best_score - second_best_person_score``. Zero when there is only one
    candidate, which is why acceptance requires the threshold test as well."""

    candidates: tuple[IdentityCandidate, ...] = field(default_factory=tuple)
    """Ranked candidates, retained for debugging and threshold calibration."""

    locked: bool = False
    """Whether temporal voting has committed to this identity for the track."""

    vote_count: int = 0
    """Consecutive agreeing votes accumulated for the winning candidate."""

    @property
    def display_name(self) -> str:
        """Label to draw on the frame: the person's name, or ``"Unknown"``."""
        if self.status is IdentityStatus.IDENTIFIED and self.name:
            return self.name
        return UNKNOWN_LABEL

    @property
    def is_identified(self) -> bool:
        """Whether a person was positively identified."""
        return self.status is IdentityStatus.IDENTIFIED

    @property
    def runner_up(self) -> IdentityCandidate | None:
        """The second-ranked candidate, if any. Useful when diagnosing margins."""
        return self.candidates[1] if len(self.candidates) > 1 else None

    def to_dict(self, *, include_candidates: bool = False) -> dict[str, Any]:
        """Return a serialisable view.

        Args:
            include_candidates: Include the full ranked list. Off by default:
                the live stream payload is per-person per-frame, and the
                candidate list is only wanted by debug and calibration views.
        """
        payload: dict[str, Any] = {
            "status": self.status.value,
            "person_id": self.person_id,
            "name": self.display_name,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "locked": self.locked,
            "vote_count": self.vote_count,
        }
        if include_candidates:
            payload["candidates"] = [c.to_dict() for c in self.candidates]
        return payload

    @classmethod
    def unknown(
        cls,
        *,
        score: float = 0.0,
        margin: float = 0.0,
        candidates: tuple[IdentityCandidate, ...] = (),
    ) -> IdentityVerdict:
        """Build a rejection verdict, preserving why it was rejected."""
        return cls(
            status=IdentityStatus.UNKNOWN,
            score=score,
            margin=margin,
            candidates=candidates,
        )

    @classmethod
    def pending(cls) -> IdentityVerdict:
        """Build a verdict for a track that has not yet been resolved."""
        return cls(status=IdentityStatus.PENDING)
