"""End-to-end verification of the Phase 1 foundation.

Answers one question — "is this installation actually working?" — without
starting a server or needing a GPU:

1. configuration loads and validates from the real ``config/`` directory
2. the required directories exist or can be created
3. structured logging writes to the JSON sink
4. the compute device resolves (reporting a fallback rather than failing)
5. the ASGI app builds and ``/api/system/health`` answers

Run after installing, or when something environmental feels wrong::

    python scripts/smoke_check.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # allow running without installing
    sys.path.insert(0, str(REPO_ROOT / "src"))

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def main() -> int:
    failures = 0

    # 1 - configuration ----------------------------------------------------
    try:
        from mcmot.core.settings import load_settings

        settings = load_settings()
        print(f"{PASS} configuration loaded  env={settings.env} version={settings.version}")
        print(f"       base_dir   {settings.paths.base_dir}")
        print(f"       database   {settings.storage.database_url}")
        print(f"       index      {settings.vector_index_file}")
    except Exception:
        print(f"{FAIL} configuration failed to load")
        traceback.print_exc()
        return 1

    # 2 - directories ------------------------------------------------------
    try:
        settings.paths.ensure_directories()
        print(f"{PASS} directories ready      data/ models/ logs/")
    except Exception:
        print(f"{FAIL} could not create the required directories")
        traceback.print_exc()
        failures += 1

    # 3 - logging ----------------------------------------------------------
    try:
        from mcmot.core.logging import configure_logging, get_logger

        configure_logging(settings.logging, log_dir=settings.paths.logs_dir, force=True)
        get_logger("smoke_check").info("smoke_check.started", check=1)

        log_file = settings.log_file_path
        if settings.logging.file_enabled:
            last = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
            json.loads(last)  # must be parseable JSON
            print(f"{PASS} logging works          {log_file}")
        else:
            print(f"{WARN} file logging disabled in configuration")
    except Exception:
        print(f"{FAIL} logging is not working")
        traceback.print_exc()
        failures += 1

    # 4 - compute device ---------------------------------------------------
    try:
        from mcmot.core.device import DeviceManager

        info = DeviceManager(settings.device).info
        label = PASS if not info.is_fallback else WARN
        print(f"{label} device {info.torch_device:<9} {info.name}")
        if info.total_vram_mb:
            print(f"       vram       {info.free_vram_mb or '?'} / {info.total_vram_mb} MiB free")
        if info.torch_version:
            print(f"       torch      {info.torch_version} (cuda {info.cuda_version or 'n/a'})")
        if info.is_fallback:
            print(f"       reason     {info.reason}")
            print("       -> install CUDA torch: .\\scripts\\setup_env.ps1 -WithTorch")
        for warning in info.warnings:
            print(f"       note       {warning}")
    except Exception:
        print(f"{FAIL} device resolution failed")
        traceback.print_exc()
        failures += 1

    # 5 - application ------------------------------------------------------
    try:
        from fastapi.testclient import TestClient

        from mcmot.main import create_app

        with TestClient(create_app(settings)) as client:
            response = client.get("/api/system/health")
            response.raise_for_status()
            payload = response.json()
        print(
            f"{PASS} api responds           "
            f"status={payload['status']} uptime={payload['uptime_s']}s"
        )
    except Exception:
        print(f"{FAIL} the application did not start")
        traceback.print_exc()
        failures += 1

    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
