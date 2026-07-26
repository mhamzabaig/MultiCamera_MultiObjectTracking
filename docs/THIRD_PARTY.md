# Third-party dependencies and licences

## Runtime (Phase 1)

| Package | Version | Licence | Why |
|---|---|---|---|
| fastapi | ~0.115 | MIT | Async HTTP, Pydantic-native validation, auto OpenAPI, streaming responses for MJPEG |
| uvicorn[standard] | ~0.32 | BSD-3 | ASGI server; `standard` adds httptools and websockets |
| pydantic | 2.x | MIT | Runtime validation for both API schemas and configuration |
| pydantic-settings | 2.x | MIT | Layered settings sources (env, dotenv, custom YAML) |
| PyYAML | 6.0.2 | MIT | Parses the YAML configuration layers |
| structlog | ~24.4 | Apache-2.0 / MIT | Structured logging with context binding |
| jinja2 | 3.1.4 | BSD-3 | Server-rendered pages |
| python-multipart | 0.0.12 | Apache-2.0 | Multipart uploads for registration images |
| numpy | >=1.26,<2.2 | BSD-3 | Array interchange between OpenCV, torch, and FAISS |

## Runtime (later phases)

| Package | Version | Licence | Phase | Note |
|---|---|---|---|---|
| torch, torchvision | 2.5.1+cu121 | BSD-3 | 3 | Installed from PyTorch's index; cu121 matches driver 537.70 (CUDA ≤ 12.2) |
| **ultralytics** | 8.3.x | **AGPL-3.0** | 3 | ⚠️ See below |
| opencv-python | 4.10 | Apache-2.0 | 7 | Decode, crop, encode, annotate |
| lap | 0.5.x | BSD-2 | 4 | Hungarian assignment for ByteTrack. Pinned explicitly because Ultralytics otherwise pip-installs it *at runtime* — see below |
| faiss-cpu | 1.9 | MIT | 2 | No `faiss-gpu` wheels for Windows; CPU search is microseconds at our scale |
| SQLAlchemy | 2.0 | MIT | 2 | Typed ORM; its dialect layer is what makes the PostgreSQL migration cheap |
| alembic | 1.14 | MIT | 2 | Versioned schema migrations |

## Vendored source

| Component | Origin | Licence | Phase |
|---|---|---|---|
| OSNet architecture (`osnet_ain_x1_0`) | [KaiyangZhou/deep-person-reid](https://github.com/KaiyangZhou/deep-person-reid) | MIT | 5 |

Vendored rather than installed: the `torchreid` package builds a Cython evaluation
extension that routinely fails on Windows, and only the model definition is
needed. The file is copied verbatim with its copyright header intact; pretrained
weights are downloaded by `scripts/download_models.py`.

## Development

| Package | Licence |
|---|---|
| pytest, pytest-asyncio, pytest-cov | MIT |
| httpx | BSD-3 |
| ruff | MIT |
| mypy | MIT |
| locust (phase 12) | MIT |

## ⚠️ Ultralytics AutoUpdate

Ultralytics runs `pip install` **at runtime** when it finds a declared-but-missing
requirement. Observed during Phase 4: it installed `lap` and numpy 2.2.6 — the
latter violating this project's `numpy<2.2` pin — into whichever site-packages it
could write to, and then crashed anyway, because a package installed mid-process is
not importable until restart.

Two mitigations, both in place:

* every Ultralytics import goes through `mcmot.ultralytics_guard.import_ultralytics`,
  which sets `YOLO_AUTOINSTALL=false` and `YOLO_OFFLINE=true` first, converting a
  silent environment mutation into a clear error naming the missing dependency;
* `lap` is pinned in the `ml` extra so the situation does not arise.

Anything new added to the `ml` extra should be checked against Ultralytics'
requirements for the same trap.

## ⚠️ Ultralytics AGPL-3.0

YOLOv11 ships under **AGPL-3.0**, which requires that anyone who interacts with a
modified version over a network be able to obtain its complete source. Practical
implications:

* **Personal, internal, or research use** — no issue.
* **Commercial product or hosted service** — you need either an
  [Ultralytics enterprise licence](https://www.ultralytics.com/license) or a
  permissively-licensed detector.

The `BaseDetector` interface exists partly to keep that swap cheap. Permissive
alternatives, in rough order of ease:

| Alternative | Licence | Note |
|---|---|---|
| RT-DETR (PaddleDetection / HF weights) | Apache-2.0 | NMS-free, strong on crowds, ~3× the latency of yolo11s |
| YOLOX | Apache-2.0 | Anchor-free, well-established, weaker than YOLOv11 |
| YOLO-NAS | custom (non-commercial weights) | Read the weight licence carefully |
| DETR / Deformable DETR | Apache-2.0 | Too slow for real-time multi-camera |

Swapping means one new file in `detection/` plus a config change; the tracker,
ReID stage, identity service, pipeline, and UI are unaffected.
