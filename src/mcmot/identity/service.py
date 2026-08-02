"""The Identity Service.

The single place that knows how a person becomes an identity and how a crop becomes
a name. Everything before it produces geometry and pixels; everything after it just
displays the answer.

Trackers never touch the database or the vector index. They ask this service, and it
owns the whole path: embed, search, aggregate, threshold, vote. That boundary is what
keeps a future distributed deployment cheap — every method here is coarse-grained and
its arguments and return values are JSON-serialisable, so putting it behind HTTP later
is an adapter rather than a redesign.

Acceptance requires **two** conditions, not one:

``score >= tau_accept``
    The match must be good in absolute terms.
``score - runner_up_score >= tau_margin``
    It must also be *clearly better than the next person*. A single threshold cannot
    express this, and it is exactly what stops a confident-but-wrong match between
    two similar-looking people. When only one person is enrolled there is no
    runner-up and the margin test is vacuous — correct, because with one identity in
    the gallery the only failure mode is a stranger, which ``tau_accept`` handles.

Write ordering is deliberate everywhere: **the database commits first, the index
second.** A crash in between leaves the index stale, which is detected by a size
comparison at startup and repaired from the database. The reverse order would leave
the index holding vectors for a person who does not exist.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from mcmot.core.errors import IdentityError
from mcmot.core.logging import get_logger
from mcmot.core.settings import IdentitySettings
from mcmot.domain.identity import (
    IdentityCandidate,
    IdentityStatus,
    IdentityVerdict,
)
from mcmot.domain.person import PersonRecord
from mcmot.domain.types import EmbeddingArray, EmbeddingMatrix, ImageArray
from mcmot.identity.voting import TrackVote, TrackVoter
from mcmot.reid.base import BaseEmbedder
from mcmot.reid.quality import QualityGate
from mcmot.storage.blobs import BlobStore
from mcmot.storage.repository import PersonRepository
from mcmot.storage.vector_index import FaissIndex

__all__ = ["IdentityService"]

_log = get_logger(__name__)

_JPEG_QUALITY = 90


class IdentityService:
    """Enrols people and resolves tracks to identities."""

    def __init__(
        self,
        *,
        config: IdentitySettings,
        embedder: BaseEmbedder,
        quality_gate: QualityGate,
        repository: PersonRepository,
        index: FaissIndex,
        blobs: BlobStore,
        voter: TrackVoter | None = None,
    ) -> None:
        """Wire the service.

        Args:
            config: Validated identity configuration.
            embedder: ReID embedder.
            quality_gate: Crop quality gate, shared with registration.
            repository: Person and embedding storage.
            index: Gallery similarity index.
            blobs: Person image store.
            voter: Vote state holder; created from ``config`` when omitted.
        """
        self._config = config
        self._embedder = embedder
        self._quality = quality_gate
        self._repo = repository
        self._index = index
        self._blobs = blobs
        self._voter = voter if voter is not None else TrackVoter(config)
        # Serialises write paths only. Reads go through the index's own lock and
        # SQLite's WAL, so identification from several cameras stays concurrent.
        self._write_lock = threading.Lock()

    # -- enrolment --------------------------------------------------------
    def enrol(
        self, name: str, crops: list[ImageArray], *, save_images: bool = True
    ) -> PersonRecord:
        """Register a new person from a set of crops.

        Crops are filtered by the quality gate, ranked, deduplicated against each
        other, and capped. Deduplication matters more than it appears: 30 frames
        captured over three seconds are mostly the same pose, and a gallery of near
        copies looks confident while covering only one viewing angle.

        Args:
            name: Display name; must be unique.
            crops: Candidate person crops in BGR ``uint8``.
            save_images: Persist representative thumbnails.

        Returns:
            The created person record.

        Raises:
            IdentityError: If too few crops survive the quality gate.
            PersonAlreadyExistsError: If the name is taken.
            StorageError: If the write fails.
        """
        registration = self._config.registration
        if not crops:
            raise IdentityError("No frames supplied for registration.")

        # 1. Quality filter, best first.
        scored = []
        rejections: dict[str, int] = defaultdict(int)
        for crop in crops:
            verdict = self._quality.assess(crop)
            if verdict.accepted:
                scored.append((verdict.score, crop))
            else:
                rejections[verdict.reason or "unknown"] += 1
        scored.sort(key=lambda item: item[0], reverse=True)

        if len(scored) < registration.min_accepted_frames:
            raise IdentityError(
                f"Only {len(scored)} of {len(crops)} frames were usable; "
                f"at least {registration.min_accepted_frames} are required. "
                "Move closer, hold steadier, and keep the whole person in frame.",
                details={
                    "supplied": len(crops),
                    "usable": len(scored),
                    "required": registration.min_accepted_frames,
                    "rejections": dict(rejections),
                },
            )

        # 2. Embed the candidates, discard anyone who is not the subject, then keep a
        #    diverse subset of what remains.
        candidates = [crop for _, crop in scored[: registration.target_frames]]
        embeddings = self._embedder.extract(candidates)

        coherent = self._coherent(embeddings)
        if len(coherent) < registration.min_accepted_frames:
            raise IdentityError(
                f"Only {len(coherent)} of {embeddings.shape[0]} frames show the same "
                f"person; at least {registration.min_accepted_frames} are required. "
                "Register with only one person in view.",
                details={
                    "embedded": int(embeddings.shape[0]),
                    "same_person": len(coherent),
                    "required": registration.min_accepted_frames,
                },
            )
        if len(coherent) < embeddings.shape[0]:
            _log.info(
                "identity.gallery_outliers_dropped",
                name=name,
                embedded=int(embeddings.shape[0]),
                kept=len(coherent),
            )

        embeddings = np.ascontiguousarray(embeddings[coherent], dtype=np.float32)
        candidates = [candidates[i] for i in coherent]
        scored = [scored[i] for i in coherent]

        keep = self._select_gallery(
            embeddings,
            limit=min(registration.max_accepted_frames, self._config.max_embeddings_per_person),
            floor=registration.min_accepted_frames,
            threshold=registration.duplicate_similarity,
        )

        chosen_vectors = np.ascontiguousarray(embeddings[keep], dtype=np.float32)
        chosen_scores = [scored[i][0] for i in keep]

        # 3. Persist: database first, index second (see the module docstring).
        with self._write_lock:
            person = self._repo.create(name)
            try:
                records = self._repo.add_embeddings(person.id, chosen_vectors, chosen_scores)
                self._index.add([r.vector_id for r in records], chosen_vectors)

                thumbnail: str | None = None
                if save_images:
                    thumbnail = self._store_images(
                        person.id,
                        [candidates[i] for i in keep][: registration.representative_images],
                    )
                    if thumbnail:
                        self._repo.set_representative_image(person.id, thumbnail)
                self._index.save()
            except Exception:
                # Roll back the whole enrolment rather than leaving a person with a
                # partial gallery, which would match badly and confuse the operator.
                self._rollback_person(person.id)
                raise

        _log.info(
            "identity.enrolled",
            person_id=person.id,
            name=person.name,
            supplied=len(crops),
            usable=len(scored),
            stored=len(keep),
            rejections=dict(rejections),
        )
        return self._repo.get(person.id)

    def _coherent(self, embeddings: EmbeddingMatrix) -> list[int]:
        """Keep only the embeddings that show the same person as the majority.

        **The failure this prevents.** Registration picks one person per frame. If a
        bystander wins a few frames — someone crossing the room behind the subject —
        those vectors join the gallery under the subject's name, and from then on the
        system recognises *both* people as that person. It looks confident while being
        wrong about someone's identity, which is the worst outcome this system has,
        and nothing downstream can detect it: a mixed gallery is internally consistent.

        **How.** Take the medoid — the embedding most similar to all the others, so
        the majority subject by construction rather than by assuming frame order — and
        drop anything that fails to match it at ``tau_accept``.

        Reusing ``tau_accept`` rather than inventing a threshold is deliberate. It is
        already the calibrated answer to "is this the same person", already tuned per
        deployment by ``scripts/calibrate.py``, and the gap it sits in is wide here:
        different views of one person score 0.93-0.99, different people around 0.55.

        Returns:
            Indices to keep, in their original order.
        """
        count = embeddings.shape[0]
        if count < 3:
            # Two frames cannot establish a majority, and rejecting one of them on the
            # other's say-so is a coin toss. The quality gate and the minimum-frame
            # floor already refuse an enrolment this thin.
            return list(range(count))

        # Rows are L2-normalised by contract, so the inner product is cosine.
        similarity = embeddings @ embeddings.T
        np.fill_diagonal(similarity, 0.0)
        medoid = int(np.argmax(similarity.sum(axis=1)))

        return [
            index
            for index in range(count)
            if index == medoid or float(similarity[medoid, index]) >= self._config.tau_accept
        ]

    def _select_gallery(
        self, embeddings: EmbeddingMatrix, *, limit: int, floor: int, threshold: float
    ) -> list[int]:
        """Choose which embeddings to store: diverse first, but never too few.

        Input is ordered best-quality first, so the greedy first pass keeps the
        sharpest example of each distinct pose.

        The second pass is the important part. Deduplication is a *preference* for
        diversity, not a hard filter, because registration frames are legitimately
        similar — thirty frames captured over three seconds while someone holds a
        phone are nearly the same pose, and OSNet scores different views of one
        person at 0.97-0.99 anyway. A strict filter therefore collapsed a real
        30-frame enrolment down to a single vector, which is exactly the fragile
        single-shot gallery multi-shot enrolment exists to avoid. So if too few
        survive, the best rejected frames are topped back up.
        """
        kept: list[int] = []
        for index in range(embeddings.shape[0]):
            if len(kept) >= limit:
                break
            vector = embeddings[index]
            if any(float(np.dot(vector, embeddings[k])) > threshold for k in kept):
                continue
            kept.append(index)

        if len(kept) >= floor:
            return kept

        chosen = set(kept)
        target = min(floor, embeddings.shape[0], limit)
        for index in range(embeddings.shape[0]):
            if len(kept) >= target:
                break
            if index not in chosen:
                kept.append(index)
                chosen.add(index)
        kept.sort()
        return kept

    def _store_images(self, person_id: str, crops: list[ImageArray]) -> str | None:
        """Write representative thumbnails; return the first one's relative path."""
        first: str | None = None
        for position, crop in enumerate(crops):
            ok, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
            if not ok:  # pragma: no cover - encoder failure is not reproducible here
                continue
            relative = self._blobs.save(person_id, f"{position:02d}.jpg", buffer.tobytes())
            if first is None:
                first = relative
        return first

    def _rollback_person(self, person_id: str) -> None:
        """Undo a partially-completed enrolment. Never raises."""
        try:
            vector_ids = self._repo.delete(person_id)
            self._index.remove(vector_ids)
            self._blobs.delete_person(person_id)
        except Exception:  # noqa: BLE001 - cleanup must not mask the original error
            _log.exception("identity.rollback_failed", person_id=person_id)

    # -- identification ---------------------------------------------------
    def should_embed(self, camera_id: str, track_id: int, frame_index: int) -> bool:
        """Whether this track is due for an embedding on this frame."""
        return self._voter.should_embed(camera_id, track_id, frame_index)

    def identify(
        self,
        camera_id: str,
        track_id: int,
        crop: ImageArray,
        *,
        frame_index: int = 0,
    ) -> IdentityVerdict:
        """Resolve a track to an identity using one crop.

        Combines a single-frame search with the track's accumulated evidence: the
        search decides this frame's candidate, and the vote decides what is
        displayed. A locked identity therefore survives individual bad frames.

        Args:
            camera_id: Camera the track belongs to.
            track_id: Tracker-assigned id.
            crop: The person crop in BGR ``uint8``.
            frame_index: Camera frame counter, for embed scheduling.

        Returns:
            The verdict to display.
        """
        self._voter.note_embedded(camera_id, track_id, frame_index)

        if self._index.size == 0:
            # Nobody is enrolled. Reported distinctly so an empty database does not
            # look like a system that fails to recognise anyone.
            return IdentityVerdict(status=IdentityStatus.DISABLED)

        quality = self._quality.assess(crop)
        if not quality.accepted:
            # Do not vote on a bad crop: a garbage embedding would count as evidence.
            existing = self._voter.get(camera_id, track_id)
            return self._verdict_from_vote(existing, reason=quality.reason)

        embedding = self._embedder.extract_one(crop)
        candidates = self.search(embedding)
        accepted = self._apply_thresholds(candidates)

        vote = self._voter.observe(camera_id, track_id, accepted)
        return self._build_verdict(vote, candidates)

    def search(self, embedding: EmbeddingArray) -> tuple[IdentityCandidate, ...]:
        """Search the gallery and aggregate hits per person, best first.

        A person is scored by the **mean of their best ``per_person_top`` hits**
        rather than by a single best match or by a centroid. A single best match
        rewards one lucky frame; a centroid blurs away the pose variety that
        multi-shot enrolment exists to capture.
        """
        query = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
        scores, ids = self._index.search(query, self._config.top_k)

        hits: list[tuple[int, float]] = [
            (int(vector_id), float(score))
            for vector_id, score in zip(ids[0], scores[0], strict=True)
            if vector_id >= 0
        ]
        if not hits:
            return ()

        owners = self._repo.owners_of([vector_id for vector_id, _ in hits])
        grouped: dict[str, list[float]] = defaultdict(list)
        names: dict[str, str] = {}
        for vector_id, score in hits:
            owner = owners.get(vector_id)
            if owner is None:
                # A vector the database no longer knows: a stale index entry. Ignored
                # rather than trusted, and the startup check will repair the index.
                continue
            person_id, name = owner
            grouped[person_id].append(score)
            names[person_id] = name

        candidates = [
            IdentityCandidate(
                person_id=person_id,
                name=names[person_id],
                score=float(
                    np.mean(sorted(person_scores, reverse=True)[: self._config.per_person_top])
                ),
                matched_embeddings=len(person_scores),
            )
            for person_id, person_scores in grouped.items()
        ]
        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return tuple(candidates)

    def _apply_thresholds(self, candidates: tuple[IdentityCandidate, ...]) -> str | None:
        """Return the accepted person id for this frame, or ``None``."""
        if not candidates:
            return None
        best = candidates[0]
        if best.score < self._config.tau_accept:
            return None
        if self._margin(candidates) < self._config.tau_margin:
            return None
        return best.person_id

    @staticmethod
    def _margin(candidates: tuple[IdentityCandidate, ...]) -> float:
        """Gap between the best candidate and the next *person*."""
        if not candidates:
            return 0.0
        if len(candidates) == 1:
            # No runner-up: the margin test cannot discriminate, so it must not veto.
            return float("inf")
        return candidates[0].score - candidates[1].score

    def _build_verdict(
        self, vote: TrackVote, candidates: tuple[IdentityCandidate, ...]
    ) -> IdentityVerdict:
        """Combine this frame's candidates with the track's accumulated vote."""
        best = candidates[0] if candidates else None
        margin = self._margin(candidates)
        finite_margin = 0.0 if margin == float("inf") else margin

        if vote.locked_person_id is not None:
            locked = next((c for c in candidates if c.person_id == vote.locked_person_id), None)
            return IdentityVerdict(
                status=IdentityStatus.IDENTIFIED,
                person_id=vote.locked_person_id,
                # Prefer the locked person's own name; fall back to the repository
                # when this frame's search did not surface them at all.
                name=locked.name if locked else self._name_of(vote.locked_person_id),
                score=locked.score if locked else 0.0,
                margin=finite_margin,
                candidates=candidates,
                locked=True,
                vote_count=vote.votes_for(vote.locked_person_id),
            )

        # Not locked. Still gathering evidence is a different answer from "stranger".
        leader = best.person_id if best else None
        gathered = vote.votes_for(leader) if leader else 0
        if gathered > 0 and gathered < self._config.vote_min_agree:
            return IdentityVerdict(
                status=IdentityStatus.PENDING,
                score=best.score if best else 0.0,
                margin=finite_margin,
                candidates=candidates,
                vote_count=gathered,
            )
        return IdentityVerdict.unknown(
            score=best.score if best else 0.0,
            margin=finite_margin,
            candidates=candidates,
        )

    def _verdict_from_vote(self, vote: TrackVote | None, *, reason: str | None) -> IdentityVerdict:
        """Answer without a usable crop, relying only on accumulated evidence."""
        if vote is not None and vote.locked_person_id is not None:
            return IdentityVerdict(
                status=IdentityStatus.IDENTIFIED,
                person_id=vote.locked_person_id,
                name=self._name_of(vote.locked_person_id),
                locked=True,
                vote_count=vote.votes_for(vote.locked_person_id),
            )
        _log.debug("identity.crop_rejected", reason=reason)
        return IdentityVerdict.pending()

    def _name_of(self, person_id: str) -> str | None:
        try:
            return self._repo.get(person_id).name
        except Exception:  # noqa: BLE001 - a deleted person is not an error here
            return None

    # -- track lifecycle --------------------------------------------------
    def forget_track(self, camera_id: str, track_id: int) -> None:
        """Discard vote state for a retired track."""
        self._voter.forget_track(camera_id, track_id)

    def retain_tracks(self, camera_id: str, track_ids: set[int]) -> int:
        """Drop vote state for tracks of this camera that are no longer live."""
        return self._voter.retain_only(camera_id, track_ids)

    # -- management -------------------------------------------------------
    def list_persons(self) -> list[PersonRecord]:
        """Return every enrolled person."""
        return self._repo.list_all()

    def get_person(self, person_id: str) -> PersonRecord:
        """Return one person.

        Raises:
            PersonNotFoundError: If no such person exists.
        """
        return self._repo.get(person_id)

    def rename_person(self, person_id: str, name: str) -> PersonRecord:
        """Change a person's display name."""
        return self._repo.rename(person_id, name)

    def delete_person(self, person_id: str) -> None:
        """Delete a person from all three stores and from live vote state.

        Raises:
            PersonNotFoundError: If no such person exists.
        """
        with self._write_lock:
            vector_ids = self._repo.delete(person_id)
            self._index.remove(vector_ids)
            self._index.save()
            self._blobs.delete_person(person_id)
        # Clear last: a live track must not keep displaying a deleted name.
        affected = self._voter.forget_person(person_id)
        _log.info(
            "identity.person_deleted",
            person_id=person_id,
            embeddings=len(vector_ids),
            tracks_affected=affected,
        )

    def rebuild_index(self) -> int:
        """Rebuild the vector index from the database and persist it.

        The recovery path for any divergence, and what backs the reload endpoint.

        Returns:
            The number of vectors indexed.
        """
        with self._write_lock:
            count = self._index.rebuild_from(self._repo.list_embeddings())
            self._index.save()
        _log.info("identity.index_rebuilt", vectors=count)
        return count

    def stats(self) -> dict[str, Any]:
        """Return gallery and vote counts for the status endpoint."""
        return {
            "persons": self._repo.count(),
            "embeddings": self._repo.count_embeddings(),
            "index_vectors": self._index.size,
            **self._voter.stats(),
        }
