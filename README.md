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

**Phases 1–11 of 12 complete — the system is usable end to end from a browser.**
Foundation, storage, detection (YOLOv11n on CUDA), tracking (ByteTrack),
re-identification (OSNet-AIN), the identity service (dual-threshold gating and
per-track temporal voting), cameras (webcam, RTSP, HTTP/MJPEG, file), the live
tracking pipeline, a 20-endpoint REST API, a seven-page web UI, and the mobile
access path.

Open the dashboard, register someone from a phone camera, point a camera at a room,
and watch names appear over the people it recognises — with strangers reported as
Unknown rather than mislabelled. Measured on an RTX 4050: **18 FPS at 55 ms per
frame**, 0.23 embeddings per frame against 4.0 for naive per-person-per-frame ReID.

Phase 12 (load tests, benchmarks, Docker, deployment docs) remains; see
[docs/roadmap.md](docs/roadmap.md).

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
python scripts\download_models.py      # fetch yolo11n.pt + osnet_ain_x1_0.pth
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

Then open <http://localhost:8000>.

| Page | Purpose |
|---|---|
| `/` | Dashboard — health, gallery, cameras, pipelines, recent activity |
| `/registration` | Enrol a person using this device's camera |
| `/tracking` | Live annotated video with per-camera metrics |
| `/database` | Browse, rename, delete; rebuild the search index |
| `/cameras` | Add and remove cameras, watch connection state |
| `/settings` `/logs` | Active configuration; structured log tail |
| `/docs` | Interactive OpenAPI documentation |

**To use your phone**, see [docs/mobile.md](docs/mobile.md). In short: tracking a
phone camera needs nothing special, but *registering* from a phone browser needs
HTTPS, because `getUserMedia` is blocked on a plain LAN address:

```powershell
.\scripts	unnel.ps1 -Install     # once
.\scripts	unnel.ps1              # prints an https:// URL for the phone
```

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
| `src/mcmot/core/` | Config, logging, errors, device. Imports nothing from the layers above |
| `src/mcmot/domain/` | Shared dataclasses (`BBox`, `Detection`, `Track`, `IdentityVerdict`) |
| `src/mcmot/storage/` | SQLite schema, repository, FAISS index, image store |
| `src/mcmot/detection/` | `BaseDetector` interface and the YOLOv11 implementation |
| `src/mcmot/tracking/` | `BaseTracker` interface and the ByteTrack implementation |
| `src/mcmot/reid/` | `BaseEmbedder`, OSNet-AIN (vendored), and the crop quality gate |
| `src/mcmot/identity/` | Enrolment, gallery search, thresholds, per-track voting |
| `src/mcmot/camera/` | Source parsing, capture threads, reconnection, camera registry |
| `src/mcmot/container.py` | Composition root — wires every layer together |
| `src/mcmot/pipeline/` | Per-camera worker, annotator, metrics, MJPEG sink |
| `src/mcmot/services/` | Use cases spanning layers (registration) |
| `src/mcmot/api/` | Routers, schemas, dependencies, middleware, error mapping |
| `src/mcmot/ui/` | Page routes, Jinja templates, stylesheet, browser scripts |
| `config/` | `default.yaml` (committed) and `local.yaml` (yours) |
| `data/` `models/` `logs/` | Generated at runtime, never committed |
| `docs/` | Architecture, roadmap, decisions |
| `tests/` | `unit/`, `api/`, later `integration/` and `load/` |

## Documentation

* [docs/architecture.md](docs/architecture.md) — design and the reasoning behind it
* [docs/roadmap.md](docs/roadmap.md) — the 12 phases, what each delivered, and every
  defect found while verifying them
* [docs/mobile.md](docs/mobile.md) — using a phone, tunnel comparison, threshold
  calibration
* [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) — dependencies and their licences

## Licensing note

Phase 3 introduces Ultralytics YOLOv11, which is **AGPL-3.0**. That is fine for
personal and internal use; a commercial product needs either an Ultralytics
enterprise licence or a permissively-licensed detector. The `BaseDetector`
interface exists partly to keep that swap cheap.
