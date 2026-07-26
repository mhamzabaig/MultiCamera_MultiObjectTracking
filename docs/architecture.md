# Architecture

## System overview

```
┌─────────────────────────── BROWSER (HTTPS) ────────────────────────────┐
│  Dashboard  Registration  Tracking  Database  Cameras  Settings  Logs  │
│  Jinja2 + Bootstrap 5 + HTMX + Alpine.js  ·  MJPEG <img> / getUserMedia│
└────────────┬───────────────────────────────────────────┬───────────────┘
      REST/JSON + multipart                        MJPEG stream
             │                                           │
┌────────────▼───────────────────────────────────────────▼───────────────┐
│                        FastAPI  (async, single process)                 │
│   api/routers: persons · cameras · tracking · system · settings · logs  │
│   DI container → services · exception→HTTP mapping · request correlation│
└────────────┬─────────────────────────────────────┬─────────────────────┘
             │                                     │
   ┌─────────▼──────────┐              ┌───────────▼────────────┐
   │  IdentityService   │◀─────────────│    CameraManager       │
   │  (thread-safe,     │   enquire    │  N × PipelineWorker    │
   │   shared by all    │──────────────▶│  (own thread each)    │
   │   cameras)         │   verdict     └───────────┬───────────┘
   └───┬────────┬───────┘                           │
       │        │                    ┌──────────────▼──────────────────┐
       │        │                    │ CaptureThread (latest-frame-wins)│
       │        │                    │   ↓ q(1)                         │
       │        │                    │ Detector → Tracker → QualityGate │
       │        │                    │   ↓ crops (batched)              │
       │        │                    │ Embedder ──▶ IdentityService     │
       │        │                    │   ↓                              │
       │        │                    │ TemporalVoter → Annotator → JPEG │
       │        │                    └──────────────┬──────────────────┘
       │        │                                   ▼
       │        │                             FrameBus (latest annotated)
       │        │
┌──────▼───┐ ┌──▼─────────────────┐   ┌────────────────────────────────┐
│ Embedder │ │  VectorIndex       │   │  ModelRegistry (singleton)      │
│  (ReID)  │ │  IndexIDMap2(      │   │  one YOLO + one OSNet instance  │
└──────────┘ │    IndexFlatIP)    │   │  shared across all cameras      │
             └──┬─────────────────┘   └────────────────────────────────┘
                │  vector_id ↔ person_id
        ┌───────▼──────────────────────────────┐
        │ Storage layer (repository pattern)   │
        │  SQLAlchemy 2.0 → SQLite (WAL)       │
        │  BlobStore → data/persons/<id>/*.jpg │
        │  Alembic migrations → PostgreSQL     │
        └──────────────────────────────────────┘
```

## The five risks this design exists to manage

**R1 — Per-frame identity decisions cause flicker.** Running ReID every frame and
labelling whoever wins produces `Hamza → Unknown → Ali → Hamza`. The fix is
structural: identity is a property of a **track**, accumulated over time with
voting and hysteresis. See [Identity resolution](#identity-resolution).

**R2 — Browser cameras require a secure context.** `getUserMedia()` is blocked on
`http://192.168.1.x:8000`; only `localhost` and HTTPS origins qualify. Registration
cannot work over plain LAN HTTP. Hence Cloudflare Tunnel, and hence a
single-origin frontend so one certificate covers pages, API, and video.

**R3 — Network streams accumulate latency.** A naive `read()` loop drains a
growing buffer, so the pipeline reasons about five-second-old frames while
reporting 25 FPS. Fixed by latest-frame-wins capture with `queue_size: 1`, not by
a larger buffer.

**R4 — Cross-domain ReID gap.** Public ReID models are trained on fixed outdoor
surveillance datasets. Enrolment crops come from a phone at chest height;
tracking crops come from a different camera, angle, and light. Benchmark rank-1
does not transfer. Mitigations: a domain-generalisable model, multi-shot
galleries, quality gating, margin thresholds, and calibration tooling.

**R5 — Two stores can diverge.** SQLite and FAISS are separate files with no
shared transaction. Rule: **SQLite is the source of truth; the vector index is a
disposable derived cache**, verified against the database at startup and
rebuildable on demand.

## Component decisions

### Detector — YOLOv11s

| Option | Latency @640 FP16 (RTX 4050) | Verdict |
|---|---|---|
| **yolo11s** | ~6–8 ms | Chosen: best accuracy/latency knee |
| yolo11n | ~4 ms | Weaker on small, distant, occluded people |
| yolo11m | ~14 ms | Halves multi-camera headroom for modest gain |
| RT-DETR-l | ~25 ms | NMS-free, strong on crowds, too heavy here |
| Grounding DINO | ~300 ms+ | Open-vocabulary; wrong tool for real time |

⚠️ Ultralytics is **AGPL-3.0**. Fine for personal and internal use; a commercial
product needs an enterprise licence or a permissive detector. The `BaseDetector`
interface exists partly to keep that swap cheap.

### Tracker — ByteTrack

ByteTrack associates on **motion only** (Kalman + IoU), plus its signature
recovery of low-confidence detections in a second association pass. It needs no
appearance model — which matters because a ReID model already runs one stage
downstream. Choosing DeepSORT/StrongSORT/BoT-SORT here would run a *second*
embedding network purely for frame-to-frame association.

| Tracker | Appearance model | Added cost | When it wins |
|---|---|---|---|
| **ByteTrack** | none | ~1 ms CPU | High FPS, static cameras, ReID handled separately |
| OC-SORT | none | ~1 ms | Non-linear motion, longer gaps |
| DeepSORT | own CNN | +8 ms | Legacy; superseded |
| StrongSORT | own CNN | +15 ms | Best IDF1, offline-ish budgets |
| BoT-SORT | optional | +5–12 ms | **Upgrade path**: camera-motion compensation, can reuse *our* embeddings |

Its known weakness — ID switches after long occlusion — does **not** corrupt
identity here. A new track ID simply re-queries the gallery and gets the same
name back, so fragmentation costs a few frames of "Unknown", not a wrong label.
`BaseTracker.update()` accepts optional embeddings so BoT-SORT can consume the
existing ReID features with no extra compute.

### ReID — OSNet-AIN x1.0

| Model | Params | GPU / batch-8 | Cross-domain | Verdict |
|---|---|---|---|---|
| **osnet_ain_x1_0** | 2.2 M | ~4 ms | Strong (instance-norm architecture search, built for domain shift) | Chosen |
| osnet_x1_0 | 2.2 M | ~4 ms | Good | Fallback |
| FastReID R50-ibn | 25 M | ~12 ms | Very strong | Heavier + Windows build friction; add later as a second backend |
| TransReID / CLIP-ReID | 86 M+ | ~45 ms | Best | Kills real-time multi-camera on a 4050 |

Given R4, a domain-generalisable 2.2M-parameter model beats a benchmark champion
that overfits its training domain. It also leaves ~4 GB VRAM for multiple streams.

`torchreid` is **not** installed as a package — it builds a Cython evaluation
extension that routinely fails on Windows, and only the model definition is
needed. The OSNet architecture is vendored (MIT, attributed in
[THIRD_PARTY.md](THIRD_PARTY.md)) and weights are fetched by a script.

The public interface is all any other module sees:

```python
class BaseEmbedder(Protocol):
    dim: int
    def extract(self, crops: list[ImageArray]) -> EmbeddingMatrix: ...
```

L2 normalisation happens **inside** the embedder. That single invariant is what
makes cosine similarity equal an inner product, which is what makes the index
choice below valid.

### Vector index — `IndexIDMap2(IndexFlatIP)`

`IndexFlatIP` is exhaustive, and that is deliberate. "Brute force" in the bad
sense means a Python loop; `IndexFlatIP` is SIMD/BLAS-vectorised exact search
that scans ~1M × 512-dim vectors in single-digit milliseconds. Our gallery is
`persons × ~15 embeddings` — hundreds to low thousands of vectors, so search
costs microseconds.

An approximate index here would only lose recall, and a missed nearest neighbour
means **a wrong name on a person's head**. Approximation is not worth taking until
exact search is a measured bottleneck.

* `IndexIDMap2` adds stable external `int64` ids, `remove_ids()` for deletion, and
  `reconstruct()` for rebuilds — none of which plain `IndexFlat` has.
* `METRIC_INNER_PRODUCT` on normalised vectors gives exact cosine, so scores are
  interpretable in `[-1, 1]` and thresholds survive a model swap.
* Scale path, config-selectable: `> 100k` vectors → `IndexHNSWFlat(M=32,
  efSearch=64)`; `> 5M` → IVF+PQ.
* Persistence: atomic temp-file + rename, sidecar JSON recording dim, metric,
  model id, and vector count. Startup compares that count against the database
  and rebuilds on mismatch (R5).

### Identity resolution

Four mechanisms, all configurable under `identity:`:

1. **Multi-shot gallery.** `K` embeddings per person, deduplicated and
   quality-filtered from enrolment. Per-person score is the **mean of that
   person's best `per_person_top` hits** — more robust than a single centroid
   (which loses pose variety) or a single max (which rewards one lucky outlier).
2. **Dual-threshold gating.** Accept only when
   `best ≥ tau_accept` **and** `best − second_best_person ≥ tau_margin`. The
   margin test is what a single threshold cannot do: block *confidently wrong*
   matches between similar-looking people.
3. **Track-level temporal voting.** Per `(camera_id, track_id)`: a sliding window
   of verdicts, an EMA of the embedding, and hysteresis — `vote_min_agree`
   agreeing votes to claim an identity, a stricter `vote_min_reject` run to lose
   it. Once locked, ReID drops to `locked_reid_interval_frames`, cutting embedding
   compute 10–20× **and** stabilising the label (R1).
4. **Quality gating.** Crops that are too small, badly proportioned,
   frame-truncated, or blurry are rejected before embedding. Bad crops do not
   merely waste GPU time — they poison the vote. The same module serves enrolment
   frame selection, so registration and inference agree on what a usable crop is.

Correct values for 2 and 3 are empirical. `scripts/calibrate.py` (phase 11) sweeps
them against your own captured data.

### Cross-camera identity

Every camera queries the same gallery, so `person_id` *is* the global identity.
The `matching/` module adds the multi-camera layer: a `(camera_id, track_id) →
global_id` registry with appearance-plus-time-window association for handoffs and
simultaneous appearances. Kept separate from `identity/` so single-camera and
multi-camera logic stay independently testable.

## Concurrency model

**One shared `ModelRegistry`, not per-camera models.** Per-camera YOLO + OSNet
would exhaust 6 GB at ~3 cameras and triple warm-up. Models load once and are
shared; GPU access is serialised by a single inference lock, which makes latency
predictable and enables cross-camera batching later.

**Threads, not processes, for v1.** OpenCV decode, PyTorch forward passes, and
JPEG encode all release the GIL in C, so threads give real parallelism here.
Processes would force frame and embedding IPC plus a second copy of every model in
VRAM. `PipelineWorker` is deliberately process-portable for when we exceed ~4
cameras.

**Identity Service is an in-process facade, not yet a microservice.** The
distributed requirement is met by *interface shape* — coarse-grained,
JSON-serialisable in and out, stateless with respect to the caller. Wrapping it in
its own FastAPI process later is a small adapter. Doing it now would add
loopback serialisation per frame per camera for no present benefit.

## Performance budget

Per 720p frame, single camera, RTX 4050:

| Stage | Cost | Technique |
|---|---|---|
| Decode | 3–6 ms | Capture thread, `CAP_PROP_BUFFERSIZE=1`, `rtsp_transport=tcp`, `q(1)` (R3) |
| YOLOv11s | 6–8 ms | FP16, `inference_mode`, `channels_last`, `cudnn.benchmark`, warm-up |
| ByteTrack | ~1 ms | CPU |
| Quality gate | <1 ms | Rejects crops before they cost GPU time |
| OSNet (batched) | 3–5 ms | One forward per frame; per-track sampling, near-zero once locked |
| Index search | <0.5 ms | Exact, microseconds at our scale |
| Annotate + encode | 3–5 ms | Separate thread, downscaled JPEG for the browser |

≈ 20–28 ms/frame → 35–50 FPS headroom for one camera; comfortable 25 FPS on 2–3
cameras. VRAM ≈ 2.5 GB of 6 GB.

## Cross-cutting concerns

**Configuration.** Layered and deep-merged: code defaults → `config/default.yaml`
→ `config/local.yaml` → `.env` → environment → explicit arguments. Every model
sets `extra="forbid"`, so a misspelled key aborts startup instead of being
silently dropped. Validation happens exactly once, at boot.

**Logging.** structlog to two sinks — pretty console, rotating JSON file — fed by
one pipeline, with stdlib loggers routed through the same formatter. The discipline
that matters: **log state changes and aggregates, never frames.** At 25 FPS across
three cameras, per-detection logging is ~75 lines/second of noise. `Throttle`
handles periodic aggregates; `bind_context` attaches correlation fields
(`request_id`, `camera_id`, `track_id`) so one line carries the context instead of
being logged repeatedly.

**Errors.** Every deliberate failure subclasses `MCMOTError` and declares a stable
`code` and `http_status` next to itself, so there is no mapping table in the API
layer to drift. Unexpected exceptions log a full traceback and return a generic
500 with a request id — a stack trace in a browser response is an
information-disclosure bug.

**Testing.** Fakes for detector/tracker/embedder returning deterministic synthetic
data, so pipeline and identity logic are testable with no GPU and no weights;
in-memory SQLite and FAISS for storage; `TestClient` for the API; a
synthetic-video integration test asserting identity stability; locust for load;
`scripts/benchmark.py` for per-stage latency.

## Decisions taken and their alternatives

| Decision | Chosen | Alternatives considered |
|---|---|---|
| Python | 3.10 (installed on the target machine) | 3.12 (spec, needs install), 3.13 (ecosystem lag) |
| Frontend | Jinja + Bootstrap + HTMX + Alpine | React/Vue/Next — rejected: second origin, Node toolchain, CORS on the hardest path |
| Phone access | Cloudflare Tunnel | Tailscale (lower latency, stays on LAN — documented fallback), self-signed (fragile on Android), plain LAN (breaks `getUserMedia`) |
| Phone as tracking camera | IP Webcam app, server pulls | Browser upload — rejected: second ingestion path, no reconnect, video through the tunnel |
| Video to browser | MJPEG | WebRTC via aiortc — deferred behind the same sink interface |
| Identity Service | In-process facade | HTTP microservice — deferred, interface already shaped for it |
| Vector index | Exact `IndexFlatIP` | HNSW / IVF+PQ — deferred, recall loss unacceptable at this scale |
