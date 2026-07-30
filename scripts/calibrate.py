"""Measure identity thresholds against your own cameras.

The defaults for ``tau_accept`` and ``tau_margin`` were measured on crops from one
photo, augmented to simulate a second camera. Real cross-camera performance is a
different question: enrolment crops come from a phone at chest height, tracking crops
come from a different camera, angle, and light, and published ReID benchmarks do not
transfer to that. This script answers it with your data.

It needs, per person, a folder of images from **two** sources:

    calibration/
      hamza/
        enrol/     frames from the registration camera (phone)
        verify/    frames of the same person from the tracking camera
      ali/
        enrol/
        verify/

Then it reports, for a sweep of thresholds:

* how often the right person is ranked first (rank-1)
* how often a genuine match is accepted (recall)
* how often a *stranger* is accepted under someone's name (false accepts)

The last one is what matters most: a missed match costs a few frames of "Unknown",
while a false accept puts the wrong name on a person's head. Strangers come for free
here — every other person's ``verify`` images are strangers to this person's gallery.

Usage::

    python scripts/calibrate.py calibration/
    python scripts/calibrate.py calibration/ --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcmot.container import Container  # noqa: E402
from mcmot.core.settings import load_settings  # noqa: E402
from mcmot.domain.types import EmbeddingMatrix, ImageArray  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
#: Thresholds swept for tau_accept. Wide enough to show where the curve turns.
ACCEPT_SWEEP = [round(0.30 + 0.025 * i, 3) for i in range(21)]
#: Thresholds swept for tau_margin.
MARGIN_SWEEP = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20]


def iter_images(folder: Path) -> Iterator[Path]:
    """Yield image files in a folder, sorted for reproducibility."""
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def load_person_crops(container: Container, folder: Path) -> tuple[list[ImageArray], int]:
    """Detect and crop the largest person in every image. Returns (crops, skipped)."""
    import cv2

    crops: list[ImageArray] = []
    skipped = 0
    for path in iter_images(folder):
        image = cv2.imread(str(path))
        if image is None:
            skipped += 1
            continue
        detections = container.detector.detect(image)
        if not detections:
            skipped += 1
            continue
        best = max(detections, key=lambda d: d.bbox.area)
        height, width = image.shape[0], image.shape[1]
        x1, y1, x2, y2 = best.bbox.clip(width, height).to_int()
        crop = image[y1:y2, x1:x2]
        if crop.size:
            crops.append(np.ascontiguousarray(crop, dtype=np.uint8))
        else:  # pragma: no cover - clip() guarantees a valid box
            skipped += 1
    return crops, skipped


def embed(container: Container, crops: list[ImageArray]) -> EmbeddingMatrix:
    """Quality-gate then embed a set of crops."""
    gate = container.quality_gate
    usable = [crop for crop in crops if gate.assess(crop).accepted]
    if not usable:
        return np.zeros((0, container.embedder.dim), dtype=np.float32)
    return container.embedder.extract(usable)


def person_score(gallery: EmbeddingMatrix, probe: np.ndarray, top: int) -> float:
    """Score one probe against one person's gallery.

    The same rule the identity service uses: the mean of that person's best ``top``
    similarities. A single max rewards one lucky frame; a centroid blurs away the
    pose variety multi-shot enrolment exists to capture.
    """
    if gallery.shape[0] == 0:
        return -1.0
    similarities = gallery @ probe
    keep = min(top, similarities.shape[0])
    return float(np.sort(similarities)[-keep:].mean())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="Folder of per-person enrol/verify sets.")
    parser.add_argument("--json", type=Path, default=None, help="Write full results here.")
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Per-person top-K aggregation. Defaults to the configured value.",
    )
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        print(f"!! {args.root} is not a directory")
        return 1

    settings = load_settings()
    container = Container(settings)
    container.startup()
    top = args.top or settings.identity.per_person_top

    try:
        # ---------------------------------------------------------- load
        people: dict[str, dict[str, Any]] = {}
        for folder in sorted(p for p in args.root.iterdir() if p.is_dir()):
            enrol_dir, verify_dir = folder / "enrol", folder / "verify"
            if not enrol_dir.is_dir() or not verify_dir.is_dir():
                print(f"   skipping {folder.name}: needs both enrol/ and verify/")
                continue

            enrol_crops, enrol_skipped = load_person_crops(container, enrol_dir)
            verify_crops, verify_skipped = load_person_crops(container, verify_dir)
            gallery = embed(container, enrol_crops)
            probes = embed(container, verify_crops)

            unusable = enrol_skipped + verify_skipped
            note = f"  ({unusable} unusable)" if unusable else ""
            print(
                f"   {folder.name:<16} gallery {gallery.shape[0]:>3} "
                f"probes {probes.shape[0]:>3}{note}"
            )
            if gallery.shape[0] == 0 or probes.shape[0] == 0:
                print(f"   !! {folder.name} has no usable crops on one side; excluded")
                continue
            people[folder.name] = {"gallery": gallery, "probes": probes}

        if len(people) < 1:
            print("!! no usable people found")
            return 1
        if len(people) < 2:
            print(
                "\n!! Only one person. Margin and false-accept numbers need at least two,\n"
                "   because a second person is what supplies the strangers."
            )

        # ---------------------------------------------- score every probe
        # For each probe: its score against its own gallery, and the best score
        # against any *other* gallery. Those two numbers are all the sweep needs.
        names = list(people)
        rows: list[dict[str, Any]] = []
        for name in names:
            for index, probe in enumerate(people[name]["probes"]):
                own = person_score(people[name]["gallery"], probe, top)
                others = {
                    other: person_score(people[other]["gallery"], probe, top)
                    for other in names
                    if other != name
                }
                best_other = max(others.values()) if others else -1.0
                best_other_name = max(others, key=lambda key: others[key]) if others else None
                rows.append(
                    {
                        "person": name,
                        "probe": index,
                        "own": own,
                        "best_other": best_other,
                        "best_other_name": best_other_name,
                        "rank1_correct": own >= best_other,
                    }
                )

        # ---------------------------------------------------------- report
        rank1 = sum(1 for row in rows if row["rank1_correct"]) / len(rows)
        own_scores = np.array([row["own"] for row in rows])
        other_scores = np.array([row["best_other"] for row in rows if row["best_other"] >= 0])

        print("\n" + "=" * 72)
        print("SEPARATION")
        print("=" * 72)
        print(f"probes                 : {len(rows)} across {len(names)} people")
        print(f"rank-1 accuracy        : {rank1:.1%}")
        print(
            f"same person   (n={len(own_scores):>3}) : "
            f"min {own_scores.min():.3f}  mean {own_scores.mean():.3f}  max {own_scores.max():.3f}"
        )
        if other_scores.size:
            print(
                f"different     (n={len(other_scores):>3}) : "
                f"min {other_scores.min():.3f}  mean {other_scores.mean():.3f}  "
                f"max {other_scores.max():.3f}"
            )
            gap = own_scores.mean() - other_scores.mean()
            print(f"separation             : {gap:+.3f}")
            if own_scores.min() > other_scores.max():
                print("                         cleanly separable — any threshold between works")
            else:
                print("                         overlapping — the sweep below is the trade-off")

        print("\n" + "=" * 72)
        print("THRESHOLD SWEEP")
        print("=" * 72)
        header = f"{'tau_accept':>10} {'tau_margin':>11} {'recall':>8}"
        print(f"{header} {'false accepts':>14} {'unknown':>9}")
        print("-" * 72)

        sweep: list[dict[str, Any]] = []
        for accept in ACCEPT_SWEEP:
            for margin in MARGIN_SWEEP:
                correct = wrong = unknown = 0
                for row in rows:
                    own, other = row["own"], row["best_other"]
                    own_ok = own >= accept and (own - max(other, 0.0)) >= margin
                    other_ok = other >= accept and (other - own) >= margin
                    if own_ok and own >= other:
                        correct += 1
                    elif other_ok:
                        wrong += 1
                    else:
                        unknown += 1
                entry = {
                    "tau_accept": accept,
                    "tau_margin": margin,
                    "recall": correct / len(rows),
                    "false_accepts": wrong / len(rows),
                    "unknown": unknown / len(rows),
                }
                sweep.append(entry)

        # Only the operating points worth reading: for each tau_accept, the smallest
        # margin that admits no false accepts. Printing 168 rows would bury it.
        for accept in ACCEPT_SWEEP:
            candidates = [
                entry
                for entry in sweep
                if entry["tau_accept"] == accept and entry["false_accepts"] == 0.0
            ]
            entry = (
                min(candidates, key=lambda item: item["tau_margin"])
                if candidates
                else min(
                    (item for item in sweep if item["tau_accept"] == accept),
                    key=lambda item: item["false_accepts"],
                )
            )
            print(
                f"{entry['tau_accept']:>10.3f} {entry['tau_margin']:>11.2f} "
                f"{entry['recall']:>7.1%} {entry['false_accepts']:>13.1%} "
                f"{entry['unknown']:>8.1%}"
            )

        # ------------------------------------------------ recommendation
        #
        # Maximum recall alone is the wrong objective when the classes separate
        # cleanly: every threshold below the stranger scores achieves 100% recall
        # with no false accepts, and picking the *lowest* of those would recommend
        # 0.30 for data whose strangers score 0.51. That leaves no room at all for a
        # stranger who happens to score higher than anyone in this sample.
        #
        # So among the threshold pairs that admit no false accepts and reach the best
        # recall, prefer the one nearest the midpoint of the observed gap — the point
        # furthest from both failure modes, which is what generalises to people this
        # sample never contained.
        clean = [entry for entry in sweep if entry["false_accepts"] == 0.0]
        best = None
        if clean:
            best_recall = max(entry["recall"] for entry in clean)
            contenders = [entry for entry in clean if entry["recall"] == best_recall]
            if other_scores.size:
                target = (float(other_scores.max()) + float(own_scores.min())) / 2.0
            else:
                # No strangers to separate from, so no gap to sit in the middle of.
                target = float(own_scores.min())
            best = min(
                contenders,
                key=lambda item: (abs(item["tau_accept"] - target), item["tau_margin"]),
            )

        print("\n" + "=" * 72)
        print("RECOMMENDATION")
        print("=" * 72)
        if best is None:
            print("No threshold pair eliminates false accepts on this data.")
            print("That usually means the enrolment and tracking views are too different.")
            print("Try: enrol from the camera you will track with, or add frames covering")
            print("more angles and distances.")
        else:
            print("Best recall with zero false accepts, placed mid-gap for headroom:\n")
            print("    identity:")
            print(f"      tau_accept: {best['tau_accept']}")
            print(f"      tau_margin: {best['tau_margin']}")
            print(
                f"\n    -> {best['recall']:.1%} of genuine matches named, "
                f"{best['unknown']:.1%} left Unknown"
            )
            if other_scores.size:
                print(
                    f"    -> sits between strangers (max {other_scores.max():.3f}) and "
                    f"matches (min {own_scores.min():.3f})"
                )
            print("\nPut that in config/local.yaml and restart.")
            current = (settings.identity.tau_accept, settings.identity.tau_margin)
            print(f"\nCurrently configured: tau_accept={current[0]}, tau_margin={current[1]}")
            if current != (best["tau_accept"], best["tau_margin"]):
                print("Those differ from the recommendation above.")

        if args.json:
            args.json.write_text(
                json.dumps(
                    {
                        "people": names,
                        "rank1": rank1,
                        "per_probe": [
                            {key: value for key, value in row.items() if key != "probe"}
                            for row in rows
                        ],
                        "sweep": sweep,
                        "recommended": best,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"\nFull results written to {args.json}")
        return 0
    finally:
        container.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
