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
| 5 | ReID | ✅ **done** | `BaseEmbedder` + OSNet-AIN + preprocessing + quality gate + batching |
| 6 | Identity Service | ✅ **done** | Enrolment, search, per-person aggregation, dual thresholds, temporal voting, CRUD, index maintenance |
| 7 | Camera | ✅ **done** | `CameraSource` (webcam / RTSP / HTTP / file), capture thread, manager, reconnection with backoff |
| 8 | Pipeline | ✅ **done** | `PipelineWorker`, frame-skip policy, state retirement, annotator, metrics, MJPEG sink |
| 9 | API | ✅ **done** | Persons, cameras, tracking routers; settings and logs endpoints; registration service; MJPEG endpoint |
| 10 | Frontend | next | Dashboard, Registration, Tracking, Database, Cameras, Settings, Logs pages |
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

## Phase 5 — what was delivered

| Module | Contents |
|---|---|
| `reid/base.py` | `BaseEmbedder`: `extract()` (batch), `extract_one()`, `warmup()`, `describe()`. L2 normalisation is part of the contract |
| `reid/osnet.py` | OSNet-AIN architecture, vendored from torchreid (MIT). Generated by transforming the upstream source, not retyped |
| `reid/embedder.py` | `OsNetEmbedder` — preprocessing, batching, normalisation, classifier head discarded |
| `reid/quality.py` | `QualityGate` — size, aspect, truncation, blur; plus a ranking score |
| `reid/__init__.py` | `create_embedder()` |
| `container.py` | `embedder` and `quality_gate` singletons |

Weights come from **the OSNet author's own Hugging Face account**
(`kaiyangzhou/osnet`), trained on MSMT17. The torchreid model zoo publishes via
Google Drive, which would have meant an extra dependency and unpredictable
rate-limiting; these are plain resumable HTTPS, fetched by
`scripts/download_models.py` alongside the detector weights.

**Does it actually work?** Real crops cut from `bus.jpg` by the real detector, then
augmented (downscale, brighten, darken, flip, blur) to simulate the same person on a
different camera:

| | min | mean | max |
|---|---|---|---|
| Same person, different conditions (n=20) | **0.773** | 0.965 | — |
| Different people (n=6) | — | 0.430 | **0.477** |

Separation **+0.296**. At the configured `tau_accept = 0.55`: 20/20 same-person pairs
accepted, 0/6 different-person pairs accepted. Architecture verified against the
checkpoint first — 552/552 tensor names match, 2.17M backbone parameters, only the
discarded classifier head differs.

**Latency is worse than predicted, and the reason matters.** ~21 ms per call, flat in
batch size, because OSNet is kernel-launch-bound rather than compute-bound. See
[architecture.md](architecture.md#reid--osnet-ain-x10) — the estimate there has been
corrected. Amortised to ~2 ms/frame by the temporal sampling already in config.

**Two defects found and fixed while building this.**

*A circular import.* `core/container.py` imported from `mcmot.reid`, which imports
back into `core`. It only manifested depending on which module a process imported
first, so every test and script passed while the cycle sat there. The container moved
to `mcmot/container.py` — the composition root belongs above the layers it wires,
not inside the lowest one. `tests/unit/test_imports.py` now imports each module in a
clean interpreter and asserts `core` pulls in no feature package.

*A useless quality score.* The ranking score returned exactly 1.000 for every real
crop, because real detections sit far above the rejection floors (blur ~4000 against
a threshold of 45) and a clipped ratio hit its ceiling immediately. Since its only
job is ranking candidate registration frames, a constant made it worthless. Replaced
with `x / (x + k)`, which saturates smoothly and preserves ordering across the whole
range; the four real crops now score 0.805–0.921.

## Phase 6 — what was delivered

| Module | Contents |
|---|---|
| `identity/voting.py` | `TrackVoter` — sliding window, asymmetric lock/unlock hysteresis, embed scheduling, track and person lifecycle |
| `identity/service.py` | `IdentityService` — enrol, identify, search + per-person aggregation, dual thresholds, CRUD, rebuild |
| `identity/__init__.py` | Exports |
| `container.py` | `identity_service` singleton — one instance, because it holds the vote state |

This is the phase where everything connects: detection produces boxes, tracking
produces continuity, ReID produces fingerprints, and this turns them into a name.

**Acceptance requires two conditions.** `score >= tau_accept` *and*
`score - runner_up >= tau_margin`. The margin test is what a single threshold cannot
express — it blocks a confident-but-wrong match between two similar-looking people.
With one person enrolled there is no runner-up and the test is vacuous by design,
since the only failure mode left is a stranger, which `tau_accept` covers.

Verified end to end against real crops, enrolling two people from `bus.jpg`:

| Property | Result |
|---|---|
| Recognition from an unseen view | Hamza 0.993, Ali 0.995, margin 0.48, both locked |
| Stranger (never enrolled) | 0.467 vs `tau_accept` 0.55 → **Unknown**, never named |
| Locked identity vs 7 conflicting frames | held as Hamza — hysteresis works |
| Locked identity vs 10 more | released, as intended |
| Genuine person switch on a reused track id | Hamza → Ali, re-locked |
| Embed scheduling | every 10 frames unlocked, every 60 once locked |
| Deleting a person | live track stopped showing the name immediately |
| Index rebuild | DB and index counts stayed equal throughout |

**A default that would have broken registration in production.**
`duplicate_similarity` defaulted to 0.97, but Phase 5 measured *different views of
the same person* at 0.97–0.99. So deduplication discarded genuine pose variety and
collapsed a 30-frame enrolment into **one** embedding — the fragile single-shot
gallery that multi-shot enrolment exists to avoid. Worse, real registration frames
(someone holding a phone for three seconds) are even more similar, so this would have
hit every user. Two changes: the threshold moved to 0.995, where only truly redundant
frames are dropped, and deduplication became a *preference* rather than a filter —
`_select_gallery` tops up from the best rejected frames when too few survive. Real
enrolment now stores 8 embeddings instead of 1.

## Phase 7 — what was delivered

| Module | Contents |
|---|---|
| `camera/source.py` | `CameraSpec` parses webcam index / RTSP / HTTP / file; `CameraSource` opens it with the right backend and properties |
| `camera/capture.py` | `CaptureThread` — latest-frame-wins, reconnect with exponential backoff, per-camera stats |
| `camera/manager.py` | `CameraManager` — one entry per camera id, open/close/latest/status |
| `container.py` | `camera_manager` singleton; cameras stop **first** at shutdown, before storage |

One class rather than four: OpenCV opens all four source kinds through `VideoCapture`,
so the differences are a handful of branches, not a hierarchy.

**Latest-frame-wins, proven rather than asserted.** A consumer running at 4 Hz against
a 25 FPS source saw frame indices `31, 37, 43, 48, 55, 61` — gaps of 5–7, i.e. frames
dropped rather than queued — with frame age at read staying between **16 and 47 ms**
throughout. A queue would have shown steadily growing staleness while still reporting a
healthy frame rate. Drops are counted (`dropped=60, delivered=61`), so the trade-off is
visible instead of hidden.

Verified: source parsing for all four kinds plus three rejection cases; capture from a
file; `take()` delivering once; file end reported as `ended` with zero reconnects;
reconnect giving up after the configured attempt limit with the reason recorded;
manager rejecting duplicate ids and leaving no entry behind after a failed open;
cameras stopping before storage at shutdown. Also against the real laptop webcam:
1280×720 as requested, 63 ms latency.

**Two defects found by running it.**

*Open timeouts were silently ignored.* `CAP_PROP_OPEN_TIMEOUT_MSEC` was being set
*after* `VideoCapture` was constructed — by which point the open has already happened.
A dead RTSP address therefore blocked for OpenCV's 30 second default rather than the
configured 1 second, which effectively breaks reconnection: a phone leaving Wi-Fi
would stall for 30 s per retry. Measured against a dead address:

| Mechanism | Time to fail |
|---|---|
| No options | 30.02 s |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS=timeout;…` | 30.03 s (no effect) |
| `OPENCV_FFMPEG_CAPTURE_OPTIONS=stimeout;…` | 30.02 s (no effect) |
| **Params passed to the constructor** | **1.09 s** |

Only the constructor-parameter form is honoured. The env var is still used, but only
for `rtsp_transport`, which has no other setter.

*File sources ran at 1812 FPS.* A file has nothing to pace it — unlike a live source,
whose `read()` blocks until the camera produces a frame — so a six-second clip was
consumed in under a second while pinning a core. That makes a video file useless as a
stand-in for a camera, and breaks the legitimate case of pointing the tracking page at
a recording. `CaptureThread._pace` now throttles non-live sources to the file's own
encoded rate, falling back to `camera.target_fps`.

## Phase 8 — what was delivered

| Module | Contents |
|---|---|
| `pipeline/worker.py` | `PipelineWorker` — one thread per camera running detect → track → sample-identify → annotate → encode; stale-frame skipping, rolling metrics, per-frame error tolerance |
| `pipeline/annotator.py` | `Annotator` — colour by verdict (green locked, amber deciding, grey stranger), shadowed text, labels clamped inside the frame |
| `pipeline/stream.py` | `mjpeg_stream` — `multipart/x-mixed-replace`, one JPEG per new frame, startup grace and stall timeout |
| `pipeline/manager.py` | `PipelineManager` — one worker per camera and the single GPU lock they share |
| `container.py` | `pipeline_manager` singleton; shutdown stops pipelines **before** cameras |

Measured on the RTX 4050 against a generated 810×1080 clip (zoom-and-pan over
`bus.jpg`, four real people), enrolling two of them from the first three seconds and
recognising over the rest:

| | Result |
|---|---|
| End to end | 17 FPS, 55 ms/frame — detect 25 ms, ReID 12 ms, annotate + encode ~18 ms |
| Coverage | 251 of 291 captured frames processed (86%), 39 unseen drops, 0 stale skips |
| Embedding load | 0.23 per frame against 4.00 for naive per-person-per-frame — **17× less** |
| Recognition | enrolled subject named at 0.99 and locked; the two never enrolled stayed **Unknown**; a newcomer showed "Identifying…" |
| First annotated frame | 0.41 s after `start()` |
| Stalled camera | 17.4 FPS while running → 5.2 after 3 s with no frames |
| Stop / restart | pipeline released, restarted clean, identities resolved again |

**The defect that mattered.** ByteTrack returns only the tracks it matched *this*
frame, but keeps a lost one internally for `track_buffer` frames and revives it under
the same id. The worker released identity state on per-frame absence, so **one missed
detection wiped a locked identity** — the label reverting to "Identifying…" and the
vote restarting for anyone who walked behind a chair. That defeats the hysteresis
phase 6 exists to provide, and it fails silently: no error, just a flickering name.
State is now retired on the tracker's own horizon, counted in *processed* frames
because that is the tracker's clock — camera frame indices advance faster whenever
frames are skipped, which would retire identities early.

**Three more found by measuring rather than reading.**

*FPS froze instead of decaying.* Computed between the first and last processed frame,
the figure never changed once a camera stalled: a dead stream kept reporting the
healthy rate it had while alive. Now measured to *now*, so it decays to zero. The
window also had to move from `time.monotonic` to `time.perf_counter` — monotonic's
resolution on Windows is ~15 ms, so several fast frames share one tick and a short
window can span exactly zero, reporting 0 FPS while running fine.

*`frames_dropped` was meaningless.* It counted "the slot was occupied", so with the
pipeline peeking via the non-consuming `latest()` it read **285 of 286** on a
perfectly healthy camera. A 100% drop rate that means nothing is worse than no
metric: it is the number that gets a real problem ignored. It now counts frames
nobody ever looked at — 39 of 291, matching the 40 the pipeline actually missed.

*One bad frame killed the camera.* Any exception ended the worker permanently. A
truncated JPEG or a transient CUDA error should cost one frame, so failures are now
tolerated up to five consecutive before the worker fails.

**Known shortfall, not yet fixed.** `_resolve_identities` calls
`IdentityService.identify()` **once per track**, so each person in a frame runs its
own OSNet forward pass. Since OSNet is kernel-launch-bound and flat in batch size
(~21 ms for 1 crop or 16), four people due on the same frame costs ~84 ms instead of
~21 ms — directly contradicting "always batch" in
[architecture.md](architecture.md#reid--osnet-ain-x10). Fixing it needs a batch entry
point on the identity service, which is a phase 6 API change; measured upside is
roughly 25% of total pipeline time on four-person frames.

Verified: `pytest` 308 passed (24 new, all with fakes — no GPU, no weights, no
camera) · `ruff` clean · `mypy --strict` clean.

## Phase 9 — what was delivered

| Module | Contents |
|---|---|
| `api/routers/persons.py` | Register (multipart), list, get, thumbnail, rename, delete, rebuild index |
| `api/routers/cameras.py` | Open, list, get, close — closing stops a running pipeline first |
| `api/routers/tracking.py` | Start, stop, list, status, and the MJPEG stream endpoint |
| `api/routers/system.py` | Adds read-only settings and a log tail with level filtering |
| `api/schemas/` | `persons`, `cameras`, `tracking` models; settings and log models on `system` |
| `services/registration.py` | `RegistrationService` — decode, detect, crop, enrol |
| `container.py` | `registration_service` singleton, sharing the tracking detector |

Twenty endpoints. **Opening a camera and tracking on it are separate operations**, so
a phone's stream URL can be checked for reachability without loading models.

**Which handlers are async.** Only the ones that await something. Everything else is a
plain `def`, which FastAPI runs in its own threadpool — so a SQLite query or a 4-second
enrolment cannot block the event loop and stall every MJPEG stream in the process.
Registration awaits the upload, then hands the CPU-and-GPU work to a worker thread.

**Settings are read-only, and that is the design.** Configuration is validated exactly
once at boot and every component holds the section it was given; mutating a threshold
mid-run would leave the process disagreeing with itself. `PUT`/`POST`/`PATCH` on the
settings endpoint return 405, asserted by a test so it stays deliberate.

Verified against a **live uvicorn server over HTTP only**, no imports of `mcmot`:

| Step | Result |
|---|---|
| Register 24 real frames | 201 in 4.0 s → 18 embeddings; index and database agreed |
| Thumbnail | 200, 44 KB, `image/jpeg` |
| Duplicate name | 409 `person.already_exists` |
| Malformed source | 422 `camera.invalid_source` |
| RTSP credentials | returned as `rtsp://***@192.0.2.9:554/s` |
| Tracking | 18.4 FPS, 55 ms latency, 3 people tracked |
| MJPEG stream | 30 chunks, 1.6 MB, 3.8 MB/s |
| Logs / settings | tail with `truncated` flag; `editable=false` |
| Teardown | stop 204, close 204, delete 204, gallery back to empty |

**Identity discrimination, measured.** Enrolling one person from a clean single-person
crop and then tracking a four-person scene: **1 named, 3 Unknown**, held across the
whole clip. The gallery separates people rather than labelling everyone.

**Two defects found by writing the tests.**

*Camera failures all claimed to be the camera's fault.* `CameraError` declares `502
Bad Gateway`, and both "that id is already open" and "that source string is nonsense"
inherited it. A browser could not tell a malformed URL from an unreachable camera —
exactly the distinction someone typing a phone's stream address needs. Split into
`InvalidCameraSourceError` (422) and `CameraAlreadyOpenError` (409).

*RTSP passwords leaked.* IP cameras are addressed as
`rtsp://user:password@host/stream`, and `CameraSpec.to_dict()` returned that string
verbatim — into the API response, into `camera.registered` in the log file, and (next
phase) onto the Cameras page. `redacted_source` now strips credentials everywhere
except the one call that must dial the camera. The settings layer's `redact_url` was
made public and shared rather than reimplemented, so the two cannot drift.

**A registration risk worth knowing about, not yet fixed.** `RegistrationService`
keeps the *largest* detection per frame, which is right for a deliberate registration
but has no coherence check on the resulting gallery. If a bystander becomes the
largest detection partway through a capture burst — someone walking behind you while
you hold the phone — the gallery silently mixes two people and both are then
recognised under the one name. This was observed while verifying: enrolling from a
four-person street scene produced a gallery that matched three different people. The
fix is outlier rejection in `_select_gallery` (keep only embeddings close to the
gallery median), which is a phase 6 change and a new threshold, so it is recorded
here rather than slipped in.

Verified: `pytest` 374 passed (66 new API tests) · `ruff` clean · `mypy --strict`
clean.

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
