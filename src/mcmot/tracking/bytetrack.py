"""ByteTrack, wrapping the Ultralytics implementation.

Why wrap rather than vendor: Ultralytics already ships a maintained ByteTrack with
its own Kalman filter and Hungarian association, and it is already installed for
the detector. Vendoring would mean copying several hundred lines of numerical code
to own and debug.

Why ByteTrack: association is motion-only (Kalman + IoU), plus its signature second
pass that recovers *low*-confidence detections which the first pass discarded — that
is what keeps a partially-occluded person's track alive. It needs no appearance
model, which matters because a ReID model already runs one stage downstream; an
appearance-based tracker would run a second embedding network purely for
frame-to-frame association.

Two upstream behaviours are corrected here:

**Track ids are remapped.** ``BYTETracker.__init__`` calls ``reset_id()``, which
resets ``BaseTrack._count`` — a *class-level global*. Starting a second camera would
therefore reset the counter mid-run and let the first camera reissue a ``track_id``
that is still alive for a different person. This wrapper hands out its own
per-instance ids, so cameras cannot interfere with each other.

**Detections are adapted.** ``BYTETracker.update`` expects an Ultralytics ``Boxes``
object. :class:`_DetectionView` supplies exactly the attributes it reads, so the
tracker never sees our domain types and we never construct a framework object.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from mcmot.core.logging import get_logger
from mcmot.core.settings import TrackerSettings
from mcmot.domain.detection import Detection
from mcmot.domain.geometry import BBox
from mcmot.domain.track import Track, TrackState
from mcmot.domain.types import BoolArray, FloatArray
from mcmot.tracking.base import BaseTracker
from mcmot.ultralytics_guard import import_ultralytics

__all__ = ["ByteTrackTracker"]

_log = get_logger(__name__)

#: Column layout of BYTETracker.update()'s return array.
_X1, _Y1, _X2, _Y2, _TRACK_ID, _SCORE, _CLS = range(7)


class _DetectionView:
    """Presents our detections in the shape ``BYTETracker.update`` expects.

    It reads ``conf``, ``xywh``, ``cls``, and ``xyxy``, needs ``len()``, and slices
    itself with a boolean mask to split high- and low-confidence detections.
    Deliberately *not* given ``xywhr``: its absence is how the tracker knows these
    are axis-aligned rather than oriented boxes.
    """

    __slots__ = ("cls", "conf", "xywh", "xyxy")

    def __init__(
        self,
        xyxy: FloatArray,
        xywh: FloatArray,
        conf: FloatArray,
        cls: FloatArray,
    ) -> None:
        self.xyxy = xyxy
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    @classmethod
    def from_detections(cls, detections: list[Detection]) -> _DetectionView:
        """Build a view from domain detections, in one vectorised pass."""
        if not detections:
            empty = np.zeros((0, 4), dtype=np.float32)
            return cls(empty, empty, np.zeros(0, np.float32), np.zeros(0, np.float32))

        xyxy = np.array([d.bbox.to_xyxy() for d in detections], dtype=np.float32)
        xywh = np.array([d.bbox.to_cxcywh() for d in detections], dtype=np.float32)
        conf = np.array([d.score for d in detections], dtype=np.float32)
        classes = np.array([d.class_id for d in detections], dtype=np.float32)
        return cls(xyxy, xywh, conf, classes)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask: BoolArray) -> _DetectionView:
        return _DetectionView(self.xyxy[mask], self.xywh[mask], self.conf[mask], self.cls[mask])


class ByteTrackTracker(BaseTracker):
    """Motion-based multi-object tracker for one camera."""

    def __init__(self, config: TrackerSettings) -> None:
        """Args: config: Validated tracker configuration."""
        self._config = config
        self._tracker = self._build()
        # Upstream id -> our id. Also carries the frame the track first appeared and
        # how many frames it has been matched, so Track.age and Track.hits are real
        # numbers rather than placeholder zeros.
        self._ids: dict[int, tuple[int, int, int]] = {}
        self._next_id = 0
        self._frame = 0

    def _build(self) -> Any:
        # Guarded import: Ultralytics would otherwise pip-install `lap` at runtime.
        byte_tracker = import_ultralytics("ultralytics.trackers.byte_tracker")

        args = SimpleNamespace(
            track_high_thresh=self._config.track_high_thresh,
            track_low_thresh=self._config.track_low_thresh,
            new_track_thresh=self._config.new_track_thresh,
            match_thresh=self._config.match_thresh,
            track_buffer=self._config.track_buffer,
            # Upstream default: weights the IoU cost by detection confidence. Not
            # exposed in config because there is no reason to turn it off.
            fuse_score=True,
        )
        return byte_tracker.BYTETracker(args, frame_rate=self._config.frame_rate)

    # -- tracking ---------------------------------------------------------
    def update(self, detections: list[Detection]) -> list[Track]:
        """Advance one frame. See :meth:`BaseTracker.update`."""
        self._frame += 1
        # Filter by area before association: tiny boxes are usually noise, and
        # feeding them in creates short-lived tracks that pollute identity voting.
        candidates = [d for d in detections if d.bbox.area >= self._config.min_box_area]

        raw = self._tracker.update(_DetectionView.from_detections(candidates))
        if raw is None or len(raw) == 0:
            self._forget_absent(set())
            return []

        tracks: list[Track] = []
        seen: set[int] = set()
        for row in np.atleast_2d(raw):
            upstream_id = int(row[_TRACK_ID])
            seen.add(upstream_id)
            track_id, first_frame, hits = self._resolve(upstream_id)
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=BBox(float(row[_X1]), float(row[_Y1]), float(row[_X2]), float(row[_Y2])),
                    score=float(min(max(row[_SCORE], 0.0), 1.0)),
                    # BYTETracker only returns activated, currently-matched tracks,
                    # so every row here is confirmed and up to date by construction.
                    state=TrackState.CONFIRMED,
                    age=self._frame - first_frame + 1,
                    hits=hits,
                    time_since_update=0,
                    class_id=int(row[_CLS]),
                )
            )

        self._forget_absent(seen)
        return tracks

    def reset(self) -> None:
        """Discard all state. See :meth:`BaseTracker.reset`."""
        self._tracker.reset()
        self._ids.clear()
        self._frame = 0
        # _next_id deliberately continues climbing: reusing an id after a reconnect
        # would let a stale identity vote attach itself to a different person.
        _log.info("tracker.reset", next_track_id=self._next_id + 1)

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the tracker."""
        return {
            "name": self._config.name,
            "match_thresh": self._config.match_thresh,
            "track_buffer": self._config.track_buffer,
            "frame_rate": self._config.frame_rate,
            "active_tracks": len(self._ids),
        }

    # -- id bookkeeping ---------------------------------------------------
    def _resolve(self, upstream_id: int) -> tuple[int, int, int]:
        """Return ``(our_id, first_frame, hits)`` for an upstream track id."""
        existing = self._ids.get(upstream_id)
        if existing is None:
            self._next_id += 1
            entry = (self._next_id, self._frame, 1)
        else:
            our_id, first_frame, hits = existing
            entry = (our_id, first_frame, hits + 1)
        self._ids[upstream_id] = entry
        return entry

    def _forget_absent(self, seen: set[int]) -> None:
        """Drop bookkeeping for tracks upstream has retired.

        Bounded by ``track_buffer``: a track absent for longer than upstream keeps
        it can never return, so holding its entry would leak memory on a stream
        that runs for days.
        """
        if not self._ids:
            return
        expiry = self._config.track_buffer
        for upstream_id, (our_id, first_frame, hits) in list(self._ids.items()):
            if upstream_id in seen:
                continue
            if self._frame - (first_frame + hits) > expiry:
                del self._ids[upstream_id]
                _log.debug("tracker.track_expired", track_id=our_id)
