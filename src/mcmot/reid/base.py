"""The person re-identification interface.

Turns a cropped person image into a fixed-length numeric fingerprint. Two crops of
the same person produce similar vectors; different people produce dissimilar ones.
That is the whole contract, and it is what lets a detected person be matched against
the enrolled gallery.

**Every returned vector is L2-normalised.** This invariant is enforced here rather
than left to callers, because it is what makes inner product identical to cosine
similarity — which is what
:class:`~mcmot.storage.vector_index.FaissIndex` relies on and what makes the
configured similarity thresholds mean the same thing regardless of which model is
loaded.

Nothing outside this package knows which model is in use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mcmot.core.errors import InvalidImageError
from mcmot.domain.types import EmbeddingArray, EmbeddingMatrix, ImageArray

__all__ = ["BaseEmbedder"]


class BaseEmbedder(ABC):
    """Produces appearance embeddings for person crops."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Embedding dimensionality. Must match ``reid.embedding_dim``."""

    @abstractmethod
    def extract(self, crops: list[ImageArray]) -> EmbeddingMatrix:
        """Embed a batch of person crops.

        Batched rather than one-at-a-time because a single forward pass over eight
        crops costs barely more than over one, and a frame routinely contains
        several people.

        Implementations must:

        * accept BGR ``uint8`` crops of any size and resize internally;
        * return shape ``(len(crops), dim)``, ``float32``, **L2-normalised**;
        * preserve input order, so ``result[i]`` belongs to ``crops[i]``;
        * return an empty ``(0, dim)`` array for an empty input.

        Args:
            crops: Person crops in BGR ``uint8``.

        Returns:
            One embedding per crop.

        Raises:
            InvalidImageError: If a crop is empty or not a BGR image.
        """

    def extract_one(self, crop: ImageArray) -> EmbeddingArray:
        """Embed a single crop.

        Concrete, not abstract: implemented once in terms of :meth:`extract` so no
        subclass can let the single and batch paths drift apart.

        Args:
            crop: Person crop in BGR ``uint8``.

        Returns:
            One L2-normalised embedding of shape ``(dim,)``.

        Raises:
            InvalidImageError: If the crop is unusable.
        """
        result = self.extract([crop])
        if result.shape[0] != 1:  # pragma: no cover - defensive
            raise InvalidImageError("Embedder returned no vector for the crop.")
        # Annotated because NumPy's stubs type integer indexing as Any.
        vector: EmbeddingArray = result[0]
        return vector

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy forward pass to absorb one-off initialisation cost."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary for the boot log and health endpoint."""
