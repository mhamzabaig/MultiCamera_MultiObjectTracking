"""The tracking pipeline: where every other module is finally joined up.

One worker per camera runs ``capture -> detect -> track -> identify -> annotate ->
encode``, publishing annotated JPEGs that the API streams to a browser as MJPEG.
"""

from __future__ import annotations

from mcmot.pipeline.annotator import Annotator
from mcmot.pipeline.manager import PipelineManager
from mcmot.pipeline.stream import MJPEG_CONTENT_TYPE, mjpeg_stream
from mcmot.pipeline.worker import PipelineState, PipelineWorker

__all__ = [
    "MJPEG_CONTENT_TYPE",
    "Annotator",
    "PipelineManager",
    "PipelineState",
    "PipelineWorker",
    "mjpeg_stream",
]
