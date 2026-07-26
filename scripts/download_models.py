"""Download model weights into ``models/``.

Idempotent: an existing file is left alone unless ``--force`` is given.

    python scripts/download_models.py
    python scripts/download_models.py --model yolo11s.pt
    python scripts/download_models.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mcmot.core.settings import load_settings  # noqa: E402

ULTRALYTICS_RELEASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"

# OSNet weights come from the model author's own Hugging Face account. The
# torchreid model zoo publishes them via Google Drive, which needs an extra
# dependency and rate-limits unpredictably; these are plain resumable HTTPS.
OSNET_REPO = "https://huggingface.co/kaiyangzhou/osnet/resolve/main"
_OSNET_AIN_X1_0 = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015"
    "_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)

KNOWN_MODELS = {
    "yolo11n.pt": f"{ULTRALYTICS_RELEASE}/yolo11n.pt",
    "yolo11s.pt": f"{ULTRALYTICS_RELEASE}/yolo11s.pt",
    "yolo11m.pt": f"{ULTRALYTICS_RELEASE}/yolo11m.pt",
    # Trained on MSMT17: larger and more varied than Market-1501, which matters
    # because enrolment and tracking here happen on different cameras.
    "osnet_ain_x1_0.pth": f"{OSNET_REPO}/{_OSNET_AIN_X1_0}",
}


def download(url: str, target: Path) -> None:
    """Fetch `url` to `target` via a temporary file, so a failure leaves no partial."""
    temp = target.with_suffix(target.suffix + ".part")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URLs
            total = int(response.headers.get("Content-Length", 0))
            written = 0
            with temp.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    written += len(chunk)
                    if total:
                        percent = written * 100 // total
                        print(f"\r  {percent:3d}%  {written >> 20} / {total >> 20} MiB", end="")
        print()
    except (urllib.error.URLError, OSError) as exc:
        temp.unlink(missing_ok=True)
        raise SystemExit(f"download failed: {exc}") from exc
    temp.replace(target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download MC-MOT model weights.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Weights file to fetch. Defaults to every model named in configuration. "
            f"Known: {', '.join(KNOWN_MODELS)}"
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args(argv)

    settings = load_settings()
    settings.paths.ensure_directories()

    # Default to everything the configuration actually needs, so a fresh checkout
    # is one command away from running rather than one command per model.
    wanted = (
        [args.model] if args.model else [settings.detector.weights.name, settings.reid.weights.name]
    )

    unknown = [name for name in wanted if name not in KNOWN_MODELS]
    if unknown:
        print(f"unknown model(s): {', '.join(unknown)}")
        print(f"known: {', '.join(KNOWN_MODELS)}")
        return 1

    for name in wanted:
        target = settings.paths.models_dir / name
        if target.is_file() and not args.force:
            print(f"already present: {name}  ({target.stat().st_size >> 20} MiB)")
            continue
        print(f"downloading {name} -> {target}")
        download(KNOWN_MODELS[name], target)
        print(f"done: {name}  ({target.stat().st_size >> 20} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
