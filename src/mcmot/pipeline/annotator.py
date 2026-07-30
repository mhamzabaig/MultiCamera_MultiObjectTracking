"""Frame annotation: draws what the system concluded onto the image.

Colour carries the verdict, because an operator reads colour before text:

* **green** — identified, and the identity is locked
* **amber** — still gathering evidence ("identifying…")
* **grey** — searched, matched nobody: a stranger

Grey rather than red for Unknown is deliberate. Red reads as an alarm, and an
unrecognised person is the *normal* case for a system that has three people enrolled
and a street outside the window. Reserving alarm colours for actual faults keeps them
meaningful.
"""

from __future__ import annotations

import cv2

from mcmot.core.settings import PipelineSettings
from mcmot.domain.identity import IdentityStatus, IdentityVerdict
from mcmot.domain.track import Track
from mcmot.domain.types import ImageArray

__all__ = ["Annotator"]

# BGR, since that is what OpenCV draws in.
_GREEN = (80, 200, 80)
_AMBER = (60, 180, 235)
_GREY = (150, 150, 150)
_SHADOW = (0, 0, 0)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Annotator:
    """Draws tracks and identity verdicts onto frames."""

    def __init__(self, config: PipelineSettings) -> None:
        """Args: config: Validated pipeline configuration."""
        self._config = config

    def draw(
        self,
        image: ImageArray,
        tracks: list[Track],
        verdicts: dict[int, IdentityVerdict],
    ) -> ImageArray:
        """Annotate a copy of the frame.

        Args:
            image: Source frame, BGR ``uint8``. Not modified.
            tracks: Tracks to draw.
            verdicts: Identity verdict per ``track_id``; a track may be absent while
                its first embedding is still pending.

        Returns:
            A new annotated frame.
        """
        canvas = image.copy()
        for track in tracks:
            verdict = verdicts.get(track.track_id)
            self._draw_track(canvas, track, verdict)
        return canvas

    # -- internals --------------------------------------------------------
    def _draw_track(
        self, canvas: ImageArray, track: Track, verdict: IdentityVerdict | None
    ) -> None:
        colour = self._colour(verdict)
        x1, y1, x2, y2 = track.bbox.clip(canvas.shape[1], canvas.shape[0]).to_int()
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

        lines = [self._name_line(verdict), f"ID {track.track_id}"]
        if verdict is not None and verdict.status is IdentityStatus.IDENTIFIED:
            lines.append(f"{verdict.score:.2f}")

        # Label above the box, or inside it when the box touches the top edge.
        scale, thickness = 0.55, 1
        sizes = [cv2.getTextSize(line, _FONT, scale, thickness)[0] for line in lines]
        heights = [size[1] + 6 for size in sizes]
        block = sum(heights)
        top = y1 - block - 4 if y1 - block - 4 > 0 else y1 + 4

        # Pull the label inside the frame. A person at either edge is exactly who an
        # operator most wants named — someone entering or leaving — so a label running
        # off the side and losing half its text is worst where it matters most.
        widest = max(size[0] for size in sizes)
        left = min(max(x1 + 2, 2), max(canvas.shape[1] - widest - 2, 2))

        y = top
        for line, height in zip(lines, heights, strict=True):
            y += height
            # Drawn twice: a dark offset copy underneath keeps text legible against
            # both a white shirt and a dark doorway.
            cv2.putText(canvas, line, (left + 1, y + 1), _FONT, scale, _SHADOW, thickness + 1)
            cv2.putText(canvas, line, (left, y), _FONT, scale, colour, thickness)

    @staticmethod
    def _colour(verdict: IdentityVerdict | None) -> tuple[int, int, int]:
        if verdict is None or verdict.status is IdentityStatus.PENDING:
            return _AMBER
        if verdict.status is IdentityStatus.IDENTIFIED:
            return _GREEN
        return _GREY

    @staticmethod
    def _name_line(verdict: IdentityVerdict | None) -> str:
        if verdict is None or verdict.status is IdentityStatus.PENDING:
            # Distinct from "Unknown": we have not decided yet, rather than decided
            # this is a stranger.
            return "Identifying..."
        if verdict.status is IdentityStatus.DISABLED:
            return "No gallery"
        return verdict.display_name
