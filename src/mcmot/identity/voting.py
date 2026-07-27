"""Per-track identity voting.

The mechanism that turns per-frame guesses into a stable label, and the single most
important piece of accuracy engineering in this system.

Without it, a track's name is whatever the current frame's embedding happened to
match best, which flickers: ``Hamza -> Unknown -> Ali -> Hamza`` across four frames
as the person turns, is briefly occluded, or walks through a shadow. Every one of
those frames produced a confident-looking similarity score; none of them is wrong
individually. Identity is simply not a property of a frame.

So identity is accumulated per ``(camera_id, track_id)``:

* a **sliding window** of recent per-frame candidates, rather than the latest one;
* **asymmetric hysteresis** — ``vote_min_agree`` observations to claim an identity,
  a larger ``vote_min_reject`` to lose one. Claiming should be responsive; giving up
  a name that has been holding steady should require real evidence, because the
  usual cause of a few bad frames is a bad viewing angle, not a different person;
* **embed scheduling**, so a track that is already locked is re-checked at a slow
  heartbeat instead of every frame. This is also the main ReID cost saving: OSNet
  costs ~21 ms per call, and sampling every tenth frame amortises it to ~2 ms.

This module holds no embeddings and touches no storage — it is pure bookkeeping over
candidate ids, which makes the policy testable and inspectable on its own.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

from mcmot.core.settings import IdentitySettings

__all__ = ["TrackVote", "TrackVoter"]


@dataclass(slots=True)
class TrackVote:
    """Accumulated identity evidence for a single track."""

    window: deque[str | None] = field(default_factory=deque)
    """Recent per-frame candidates, newest last. ``None`` records a frame that
    matched nobody, which is evidence too — a run of them is how a locked identity
    is eventually released."""

    locked_person_id: str | None = None
    """The committed identity, or ``None`` while still undecided."""

    last_embed_frame: int | None = None
    """Frame index of the most recent embedding, for scheduling the next one."""

    observations: int = 0
    """Total observations ever recorded, for diagnostics."""

    @property
    def is_locked(self) -> bool:
        """Whether an identity has been committed for this track."""
        return self.locked_person_id is not None

    def votes_for(self, person_id: str | None) -> int:
        """Number of observations in the current window naming ``person_id``."""
        return sum(1 for entry in self.window if entry == person_id)


class TrackVoter:
    """Maintains vote state for every live track across every camera.

    Thread-safe: several camera pipelines call into this concurrently, and each
    operation is a short critical section over a dict.
    """

    def __init__(self, config: IdentitySettings) -> None:
        """Args: config: Validated identity configuration."""
        self._config = config
        self._votes: dict[tuple[str, int], TrackVote] = {}
        self._lock = threading.Lock()

    # -- scheduling -------------------------------------------------------
    def should_embed(self, camera_id: str, track_id: int, frame_index: int) -> bool:
        """Whether this track should be embedded on this frame.

        A track is embedded immediately on first sight, then every
        ``reid_interval_frames``, dropping to ``locked_reid_interval_frames`` once
        locked. The heartbeat continues after locking rather than stopping entirely,
        so a track that silently switched to a different person — an ID switch the
        tracker did not notice — is eventually caught.

        Args:
            camera_id: Camera the track belongs to.
            track_id: Tracker-assigned id, unique per camera.
            frame_index: Monotonic frame counter for this camera.

        Returns:
            True when an embedding should be computed for this track now.
        """
        with self._lock:
            vote = self._votes.get((camera_id, track_id))
            if vote is None or vote.last_embed_frame is None:
                return True
            interval = (
                self._config.locked_reid_interval_frames
                if vote.is_locked
                else self._config.reid_interval_frames
            )
            return (frame_index - vote.last_embed_frame) >= interval

    def note_embedded(self, camera_id: str, track_id: int, frame_index: int) -> None:
        """Record that this track was embedded on this frame.

        Separate from :meth:`observe` because a crop can be embedded and still yield
        no candidate; the schedule must advance either way, or a track whose crops
        keep failing would be re-embedded on every single frame.
        """
        with self._lock:
            self._ensure(camera_id, track_id).last_embed_frame = frame_index

    # -- voting -----------------------------------------------------------
    def observe(self, camera_id: str, track_id: int, person_id: str | None) -> TrackVote:
        """Record one per-frame candidate and re-evaluate the lock.

        Args:
            camera_id: Camera the track belongs to.
            track_id: Tracker-assigned id.
            person_id: Best candidate for this frame, or ``None`` for no match.

        Returns:
            The updated vote state, including the committed identity if any.
        """
        with self._lock:
            vote = self._ensure(camera_id, track_id)
            vote.window.append(person_id)
            vote.observations += 1
            self._reevaluate(vote)
            return vote

    def _reevaluate(self, vote: TrackVote) -> None:
        """Apply the lock and unlock rules to a vote's current window."""
        counts = Counter(entry for entry in vote.window if entry is not None)

        if not vote.is_locked:
            if not counts:
                return
            leader, tally = counts.most_common(1)[0]
            if tally >= self._config.vote_min_agree:
                vote.locked_person_id = leader
            return

        # Already locked: only a sustained run of disagreement releases it.
        disagreements = sum(1 for entry in vote.window if entry != vote.locked_person_id)
        if disagreements < self._config.vote_min_reject:
            return

        vote.locked_person_id = None
        # Re-lock immediately if the window already favours somebody else, so a
        # genuine ID switch resolves in one step instead of spending
        # `vote_min_agree` more frames as Unknown.
        if counts:
            leader, tally = counts.most_common(1)[0]
            if tally >= self._config.vote_min_agree:
                vote.locked_person_id = leader

    # -- lifecycle --------------------------------------------------------
    def forget_track(self, camera_id: str, track_id: int) -> None:
        """Discard state for a retired track.

        Called when the tracker drops a track. Without it, a day-long stream leaks
        one entry per person who ever appeared.
        """
        with self._lock:
            self._votes.pop((camera_id, track_id), None)

    def retain_only(self, camera_id: str, track_ids: set[int]) -> int:
        """Drop state for every track of this camera not in ``track_ids``.

        The bulk form of :meth:`forget_track`, for a pipeline that knows which
        tracks are still live but not which ones disappeared.

        Returns:
            The number of entries dropped.
        """
        with self._lock:
            stale = [key for key in self._votes if key[0] == camera_id and key[1] not in track_ids]
            for key in stale:
                del self._votes[key]
            return len(stale)

    def forget_person(self, person_id: str) -> int:
        """Remove a deleted person from every track's accumulated evidence.

        Called when a person is deleted, so a live track cannot keep displaying a
        name that no longer exists in the database.

        Returns:
            The number of tracks affected.
        """
        with self._lock:
            affected = 0
            for vote in self._votes.values():
                if person_id not in vote.window and vote.locked_person_id != person_id:
                    continue
                affected += 1
                vote.window = deque(
                    (None if entry == person_id else entry for entry in vote.window),
                    maxlen=vote.window.maxlen,
                )
                if vote.locked_person_id == person_id:
                    vote.locked_person_id = None
                self._reevaluate(vote)
            return affected

    def clear(self) -> None:
        """Discard all vote state."""
        with self._lock:
            self._votes.clear()

    # -- introspection ----------------------------------------------------
    def get(self, camera_id: str, track_id: int) -> TrackVote | None:
        """Return the vote state for a track, or ``None`` if untracked."""
        with self._lock:
            return self._votes.get((camera_id, track_id))

    def stats(self) -> dict[str, Any]:
        """Return counts for the tracking status endpoint."""
        with self._lock:
            return {
                "tracked": len(self._votes),
                "locked": sum(1 for v in self._votes.values() if v.is_locked),
            }

    # -- helpers ----------------------------------------------------------
    def _ensure(self, camera_id: str, track_id: int) -> TrackVote:
        """Return the vote for a track, creating it on first sight. Caller holds the lock."""
        key = (camera_id, track_id)
        vote = self._votes.get(key)
        if vote is None:
            vote = TrackVote(window=deque(maxlen=self._config.vote_window))
            self._votes[key] = vote
        return vote
