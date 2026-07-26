# Roadmap

Twelve phases, built one at a time. Each ships folder structure, code,
explanation, tests, and integration instructions, and is reviewed before the next
begins.

| # | Phase | Status | Delivers |
|---|---|---|---|
| 1 | Foundation | ✅ **done** | Skeleton, layered config, structured logging, exception hierarchy, device manager, domain contracts, DI container, API skeleton |
| 2 | Storage | ✅ **done** | Two SQLite tables, `PersonRepository`, `FaissIndex` with atomic persistence, blob store, startup consistency check and rebuild |
| 3 | Detection | ✅ **done** | `BaseDetector` + `YoloDetector` (yolo11n) + factory, warm-up, FP16, weight download script |
| 4 | Tracking | ✅ **done** | `BaseTracker` + `ByteTrackTracker` + factory, per-camera id isolation |
| 5 | ReID | next | `BaseEmbedder` + OSNet + preprocessing + quality gate + batching |
| 6 | Identity Service | | Enrolment, search, per-person aggregation, dual thresholds, temporal voting, CRUD, index maintenance |
| 7 | Camera | | `CameraSource` (webcam / RTSP / HTTP / MJPEG / file), capture thread, manager, reconnection with backoff |
| 8 | Pipeline | | `PipelineWorker`, frame-skip policy, frame bus, annotator, metrics, MJPEG sink |
| 9 | API | | Persons, cameras, tracking, settings, logs routers; DI wiring; error handlers |
| 10 | Frontend | | Dashboard, Registration, Tracking, Database, Cameras, Settings, Logs pages |
| 11 | Mobile / HTTPS | | Cloudflare Tunnel scripts, phone-as-camera setup, threshold calibration tooling |
| 12 | Hardening | | Full unit / integration / API / load suites, benchmarks, Docker, deployment docs |

Phases 1–2 are load-bearing: every later phase plugs into interfaces defined
there.

## Phase 1 — what was delivered

| Module | Contents |
|---|---|
| `core/settings.py` | Layered config, 11 validated sections, `{data_dir}` expansion, weight-path anchoring, credential redaction |
| `core/logging.py` | structlog dual sink, context binding, `Throttle` for aggregates |
| `core/errors.py` | 13 exception types with stable codes and HTTP statuses |
| `core/device.py` | CUDA probing, VRAM gating, FP16 capability check, CPU fallback, warm-up |
| `core/container.py` | Composition root, lifecycle, health snapshot |
| `domain/` | `BBox`, `Detection`, `Track`, `Frame`, `IdentityVerdict`, `PersonRecord`, `EmbeddingRecord`, typed array aliases |
| `api/` | Routers, schemas, DI providers, request correlation, RFC 7807 error mapping |
| `tests/` | 208 tests across config, logging, errors, device, domain, container, API |

Verified: `pytest` 208 passed · `ruff check` clean · `ruff format` clean · `mypy
--strict` clean · `scripts/smoke_check.py` all checks passed.

## Phase 2 — what was delivered

| Module | Contents |
|---|---|
| `storage/models.py` | `persons` and `embeddings`. Vectors as `float32` BLOBs; `vector_id` doubles as the FAISS id |
| `storage/db.py` | Engine, sessions with commit/rollback, four SQLite PRAGMAs (`foreign_keys`, WAL, `synchronous`, `busy_timeout`) |
| `storage/repository.py` | `PersonRepository`: create, get, list, rename, delete, add/list/delete embeddings, `owners_of` |
| `storage/vector_index.py` | `FaissIndex`: `IndexIDMap2(IndexFlatIP)`, atomic save, dimension-checked load, `rebuild_from` |
| `storage/blobs.py` | Per-person image files with path-traversal guards |
| `core/container.py` | Storage wiring plus `_prepare_vector_index()` — the startup consistency check |
| `scripts/init_db.py` | Idempotent schema and index initialisation, `--rebuild-index` |

Kept out deliberately, per review: **no Alembic** (`create_all()` suffices until a
shipped schema must change), **no index abstraction or factory** (one concrete
class), **no separate vector file store** (BLOBs in the one source of truth), and
the unused `hnsw_*` config keys were **removed** rather than left as dead options.

Verified: `pytest` 269 passed · `ruff` clean · `mypy --strict` clean · a real
round trip on disk — enrol 2 people with 10 embeddings, search (self-match 1.0000,
stranger 0.0591), delete one across both stores, reopen from disk, then wipe the
index and confirm it heals from the database on the next startup.

## Phase 3 — what was delivered

| Module | Contents |
|---|---|
| `detection/base.py` | `BaseDetector`: `detect()`, `warmup()`, `describe()`. No framework types cross this boundary |
| `detection/yolo.py` | `YoloDetector` — Ultralytics confined to one file; FP16 gated on real hardware capability |
| `detection/__init__.py` | `create_detector()` — a dict lookup, not a plugin registry |
| `scripts/download_models.py` | Fetches `yolo11n.pt` / `yolo11s.pt` / `yolo11m.pt` atomically |
| `core/container.py` | `detector` singleton, warmed up on first access |

Default weights are **yolo11n**, per review. No test file for this phase, also per
review — verified manually instead (below).

Measured on the RTX 4050 (yolo11n, FP16, `imgsz=640`):

| Input | Latency | Throughput |
|---|---|---|
| 810×1080 | 15.1 ms | 66 FPS |
| 1280×720 | 13.8 ms | 72 FPS |

On `bus.jpg`: 4 people found (0.888 / 0.879 / 0.856 / 0.622), boxes tight and in
input-frame coordinates, bus correctly ignored, score-sorted; grayscale and float32
inputs rejected; a blank frame yields zero detections rather than an error.

**A default corrected by real data.** Those detections had aspect ratios of
0.22–0.38, but `reid.quality.min_aspect_ratio` defaulted to 0.25 — which would have
silently dropped the partially-occluded person at the frame edge (0.215) before ReID
ever saw them. Lowered to 0.18: shape is now a loose sanity check, and blur and
truncation do the real filtering.

## Phase 4 — what was delivered

| Module | Contents |
|---|---|
| `tracking/base.py` | `BaseTracker`: `update()`, `reset()`, `describe()` |
| `tracking/bytetrack.py` | `ByteTrackTracker` wrapping Ultralytics' ByteTrack, plus `_DetectionView` |
| `tracking/__init__.py` | `create_tracker()` |
| `ultralytics_guard.py` | Guarded import shared by detector and tracker (see below) |
| `core/container.py` | `create_tracker()` as a **method**, not a singleton — trackers are per-camera stateful |

Wrapped rather than vendored: Ultralytics already ships a maintained ByteTrack and
was already installed, so vendoring meant owning several hundred lines of Kalman
and Hungarian-assignment code. No test file, per review; verified manually.

**Two upstream defects corrected.**

*Track ids.* `BYTETracker.__init__` calls `reset_id()`, which resets
`BaseTrack._count` — a class-level **global**. Starting a second camera would reset
the counter mid-run and let the first camera reissue a `track_id` still alive for a
different person. The wrapper hands out its own per-instance ids; the same dict
supplies real `age` and `hits` instead of placeholder zeros.

*Runtime pip installs.* Ultralytics' AutoUpdate **ran pip during inference**,
installing `lap` and numpy 2.2.6 (violating this project's `numpy<2.2` pin) into
whatever site-packages it could write to — and then still crashed, because the new
package is not importable until the process restarts. `ultralytics_guard` sets
`YOLO_AUTOINSTALL=false` and `YOLO_OFFLINE=true` before every Ultralytics import,
and `lap` is now pinned in the `ml` extra.

Verified on the RTX 4050:

| Check | Result |
|---|---|
| Two people crossing, 40 frames | 2 stable ids, no switch (40/40 frames each) |
| Empty frames | tracks expire after `track_buffer` |
| Second camera started mid-run | no duplicate ids within a camera |
| Real detect → track, 60-frame pan | 57 FPS end to end; longest track survived 60/60 |
| `reset()` | ids never reused |

## Deliberate deferrals

Recorded so they are choices rather than oversights.

| Deferred | Until | Why |
|---|---|---|
| WebRTC video transport | after phase 10 | MJPEG is ~30 lines and works everywhere; WebRTC needs signalling and ICE for a modest LAN gain. Sink interface already allows the swap. |
| Identity Service over HTTP | when a second host exists | Interface is already coarse-grained and serialisable; a loopback hop now costs latency for no benefit. |
| Multiprocessing pipeline | beyond ~4 cameras | Decode, inference, and encode release the GIL; processes would duplicate models in 6 GB of VRAM. |
| Approximate vector index | > 100k vectors | Exact search costs microseconds at our scale and recall loss means wrong names. |
| PostgreSQL | on deployment need | SQLAlchemy 2.0 + Alembic + no SQLite-specific types keeps the migration a storage-layer change. |
| Tailscale access path | optional alternative | Cloudflare Tunnel chosen; a Tailscale profile is a config entry, not a redesign. |
| FastReID / TransReID backends | when accuracy demands it | `BaseEmbedder` allows a second backend without touching the pipeline. |
