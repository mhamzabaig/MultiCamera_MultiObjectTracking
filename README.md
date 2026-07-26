# MC-MOT

Multi-Camera Multi-Object Tracking and Person Re-Identification platform.

Detects people, tracks them in real time, identifies them against an enrolled
gallery, and recognises the same person across multiple cameras.

```
Video stream → YOLOv11 → ByteTrack → crop → OSNet ReID → Identity Service → dashboard
                                                              ├── FAISS index
                                                              └── SQLite (source of truth)
```

## Status

**Phases 1–4 of 12 complete.** Foundation (configuration, structured logging,
exception hierarchy, device management, domain contracts, DI container, API
skeleton), storage (SQLite, `PersonRepository`, FAISS gallery index, image store),
detection (YOLOv11n on CUDA — 72 FPS at 720p on an RTX 4050), and tracking
(ByteTrack — 57 FPS detect-and-track end to end). ReID, the identity service, and
the UI arrive in later phases; see [docs/roadmap.md](docs/roadmap.md).

## Requirements

* Windows 11 (Linux works; only the helper scripts are PowerShell)
* Python 3.10
* Optional: an NVIDIA GPU with driver ≥ 527 for CUDA 12.1. Without one the system
  runs on CPU and reports itself as `degraded`.

## Setup

```powershell
.\scripts\setup_env.ps1 -WithStorage -WithTorch -WithMl
.\.venv\Scripts\Activate.ps1
python scripts\init_db.py              # create the schema and an empty index
python scripts\download_models.py      # fetch yolo11n.pt
python scripts\smoke_check.py          # verify config, logging, device, API
```

The CUDA PyTorch wheel is a **2.3 GB** download from PyTorch's own index. It may
need a few attempts on an unstable link; pip resumes where it left off, so simply
re-run the script. Omit `-WithTorch -WithMl` for a CPU-only install — everything
still runs, just slowly, and the device manager reports itself as `degraded`.

## Running

```powershell
python -m mcmot.main                   # http://localhost:8000
python -m mcmot.main --reload          # auto-reload during development
```

| Endpoint | Purpose |
|---|---|
| `/api/system/health` | Status, resolved device, effective configuration |
| `/api/system/version` | Build identity (cheap, for uptime polling) |
| `/docs` | Interactive OpenAPI documentation |

## Configuration

Layered, lowest priority first:

```
code defaults → config/default.yaml → config/local.yaml → .env → environment
```

Layers are deep-merged, so a local override of `identity.tau_accept` leaves the
rest of that section alone. For machine-specific values:

```powershell
Copy-Item config\local.example.yaml config\local.yaml
```

Any value is also settable as an environment variable:

```powershell
$env:MCMOT__IDENTITY__TAU_ACCEPT = "0.62"
$env:MCMOT__DEVICE__PREFER = "cpu"
```

Unknown `MCMOT__*` keys and unknown YAML keys **abort startup**. A silently
ignored threshold is a bug that surfaces weeks later as a wrong name on a frame.

## Development

```powershell
pytest                    # full suite
pytest tests\unit -q      # fast feedback
ruff check . ; ruff format --check .
mypy
```

## Layout

| Path | Contents |
|---|---|
| `src/mcmot/core/` | Config, logging, errors, device, DI container |
| `src/mcmot/domain/` | Shared dataclasses (`BBox`, `Detection`, `Track`, `IdentityVerdict`) |
| `src/mcmot/storage/` | SQLite schema, repository, FAISS index, image store |
| `src/mcmot/detection/` | `BaseDetector` interface and the YOLOv11 implementation |
| `src/mcmot/tracking/` | `BaseTracker` interface and the ByteTrack implementation |
| `src/mcmot/api/` | Routers, schemas, dependencies, middleware, error mapping |
| `config/` | `default.yaml` (committed) and `local.yaml` (yours) |
| `data/` `models/` `logs/` | Generated at runtime, never committed |
| `docs/` | Architecture, roadmap, decisions |
| `tests/` | `unit/`, `api/`, later `integration/` and `load/` |

## Documentation

* [docs/architecture.md](docs/architecture.md) — design and the reasoning behind it
* [docs/roadmap.md](docs/roadmap.md) — the 12 phases and their order
* [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) — dependencies and their licences

## Licensing note

Phase 3 introduces Ultralytics YOLOv11, which is **AGPL-3.0**. That is fine for
personal and internal use; a commercial product needs either an Ultralytics
enterprise licence or a permissively-licensed detector. The `BaseDetector`
interface exists partly to keep that swap cheap.
