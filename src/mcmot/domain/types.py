"""Typed NumPy aliases used across the pipeline.

Bare ``np.ndarray`` says nothing about layout, and this system moves several
incompatible kinds of array around: BGR frames, normalised embeddings, and
stacked embedding matrices. Naming them makes intent checkable — a signature
taking an :data:`ImageArray` cannot silently be handed a float embedding — and
keeps ``mypy --strict`` satisfied without blanket ``Any``.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "BoolArray",
    "EmbeddingArray",
    "EmbeddingMatrix",
    "FloatArray",
    "IdArray",
    "ImageArray",
    "ScoreArray",
]

ImageArray: TypeAlias = npt.NDArray[np.uint8]
"""BGR image of shape ``(height, width, 3)`` — OpenCV's native layout."""

EmbeddingArray: TypeAlias = npt.NDArray[np.float32]
"""A single L2-normalised embedding of shape ``(dim,)``."""

EmbeddingMatrix: TypeAlias = npt.NDArray[np.float32]
"""A batch of embeddings of shape ``(n, dim)``, as FAISS and ReID batching want."""

ScoreArray: TypeAlias = npt.NDArray[np.float32]
"""Similarity or confidence scores of shape ``(n,)``."""

IdArray: TypeAlias = npt.NDArray[np.int64]
"""Vector ids returned by an index search. ``-1`` marks an empty result slot."""

FloatArray: TypeAlias = npt.NDArray[np.float32]
"""General ``float32`` array — box coordinates, confidences, tracker interchange."""

BoolArray: TypeAlias = npt.NDArray[np.bool_]
"""Boolean mask used to select rows of another array."""
