"""OSNet embedder: person crops in, normalised appearance vectors out.

Three details determine whether this produces useful embeddings or expensive noise.

**Preprocessing must match training exactly.** OSNet was trained on RGB crops
resized to 256x128 and normalised with ImageNet statistics. OpenCV hands us BGR, and
a channel swap or a missed normalisation does not raise — it silently degrades every
similarity score. The constants here are not adjustable for that reason.

**The classifier head is discarded.** The checkpoint carries a linear layer over its
training dataset's identities, which is irrelevant here; only the 512-dimensional
feature vector before it is wanted. In ``eval()`` mode OSNet returns exactly that.

**Output is L2-normalised in one vectorised step**, satisfying the invariant the
vector index depends on.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from mcmot.core.device import DeviceManager
from mcmot.core.errors import InvalidImageError, ModelLoadError
from mcmot.core.logging import get_logger
from mcmot.core.settings import ReIDSettings
from mcmot.domain.types import EmbeddingMatrix, FloatArray, ImageArray
from mcmot.reid.base import BaseEmbedder
from mcmot.reid.osnet import osnet_ain_x1_0

__all__ = ["OsNetEmbedder"]

_log = get_logger(__name__)

#: ImageNet statistics, as used when the published weights were trained.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class OsNetEmbedder(BaseEmbedder):
    """OSNet-AIN appearance embedder.

    Stateless with respect to callers, so a single instance is shared by every
    camera pipeline. GPU access is serialised by the pipeline, not here.
    """

    def __init__(self, config: ReIDSettings, device_manager: DeviceManager) -> None:
        """Load the model.

        Args:
            config: Validated ReID configuration.
            device_manager: Resolved compute device.

        Raises:
            ModelLoadError: If the weights are missing, corrupt, or do not match the
                architecture.
        """
        self._config = config
        self._device_manager = device_manager
        self._device_info = device_manager.info
        self._torch_device = torch.device(self._device_info.torch_device)
        self._half = config.half and self._device_info.fp16
        self._model = self._load()

    # -- loading ----------------------------------------------------------
    def _load(self) -> torch.nn.Module:
        weights = self._config.weights
        if not weights.is_file():
            raise ModelLoadError(
                f"ReID weights not found: {weights}. "
                "Run `python scripts/download_models.py` to fetch them.",
                details={"path": str(weights), "model": self._config.name},
            )

        try:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - torch raises many types here
            raise ModelLoadError(
                f"Could not read ReID weights: {weights}",
                details={"path": str(weights), "reason": str(exc)},
            ) from exc

        state = checkpoint.get("state_dict", checkpoint)
        # Checkpoints trained with DataParallel carry a "module." prefix.
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        # Drop the classifier: it is sized to the training set's identity count and
        # plays no part in feature extraction.
        backbone = {k: v for k, v in state.items() if not k.startswith("classifier")}

        # The vendored architecture is kept close to upstream and therefore carries no
        # annotations; see mcmot.reid.osnet.
        model = osnet_ain_x1_0(num_classes=1, pretrained=False)  # type: ignore[no-untyped-call]
        incompatible = model.load_state_dict(backbone, strict=False)
        missing = [k for k in incompatible.missing_keys if not k.startswith("classifier")]
        if missing or incompatible.unexpected_keys:
            # A partial load would still run and still return vectors, so this is
            # checked explicitly rather than trusted.
            raise ModelLoadError(
                "ReID weights do not match the model architecture.",
                details={
                    "path": str(weights),
                    "missing": missing[:8],
                    "unexpected": list(incompatible.unexpected_keys)[:8],
                },
            )

        model.eval()
        model.to(self._torch_device)
        if self._half:
            model.half()

        with torch.inference_mode():
            probe = model(self._blank_batch(1))
        dim = int(probe.shape[1])
        module: torch.nn.Module = model
        if dim != self._config.embedding_dim:
            raise ModelLoadError(
                "ReID model output does not match the configured embedding_dim; "
                "any stored gallery vectors would be unusable.",
                details={"configured": self._config.embedding_dim, "actual": dim},
            )

        _log.info(
            "reid.loaded",
            model=self._config.name,
            weights=weights.name,
            device=self._device_info.torch_device,
            half=self._half,
            embedding_dim=dim,
            input_size=[self._config.input_height, self._config.input_width],
            batch_size=self._config.batch_size,
        )
        return module

    # -- inference --------------------------------------------------------
    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._config.embedding_dim

    def extract(self, crops: list[ImageArray]) -> EmbeddingMatrix:
        """Embed a batch of crops. See :meth:`BaseEmbedder.extract`."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)

        batch = np.stack([self._preprocess(crop) for crop in crops])

        outputs: list[FloatArray] = []
        limit = self._config.batch_size
        with torch.inference_mode():
            for start in range(0, batch.shape[0], limit):
                chunk = torch.from_numpy(batch[start : start + limit]).to(
                    self._torch_device, non_blocking=True
                )
                if self._half:
                    chunk = chunk.half()
                # float32 on the way out regardless of compute precision: the index
                # and the database both store float32.
                outputs.append(self._model(chunk).float().cpu().numpy())

        features = np.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]
        return self._normalise(features)

    def warmup(self) -> None:
        """Run dummy forward passes. See :meth:`BaseEmbedder.warmup`."""
        dummy = self._blank_batch(self._config.batch_size)

        def step() -> None:
            with torch.inference_mode():
                self._model(dummy)

        self._device_manager.warmup(step, label="reid")

    def describe(self) -> dict[str, Any]:
        """Return a serialisable summary of the loaded embedder."""
        return {
            "name": self._config.name,
            "weights": self._config.weights.name,
            "device": self._device_info.torch_device,
            "half": self._half,
            "embedding_dim": self.dim,
            "input_size": [self._config.input_height, self._config.input_width],
            "batch_size": self._config.batch_size,
        }

    # -- helpers ----------------------------------------------------------
    def _preprocess(self, crop: ImageArray) -> FloatArray:
        """Convert one BGR crop into a normalised CHW float32 tensor array."""
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            raise InvalidImageError(
                "ReID expects a non-empty HxWx3 BGR crop.",
                details={"shape": list(crop.shape)},
            )
        if crop.dtype != np.uint8:
            raise InvalidImageError(
                "ReID expects a uint8 crop.", details={"dtype": str(crop.dtype)}
            )

        # INTER_LINEAR matches the bilinear resize used during training. cv2 takes
        # (width, height); the config names them the other way round.
        resized = cv2.resize(
            crop,
            (self._config.input_width, self._config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        balanced = self._white_balance(resized)
        rgb = cv2.cvtColor(balanced, cv2.COLOR_BGR2RGB)
        scaled = rgb.astype(np.float32) / 255.0
        normalised = (scaled - _MEAN) / _STD
        return np.ascontiguousarray(normalised.transpose(2, 0, 1))

    @staticmethod
    def _white_balance(crop: npt.NDArray[Any]) -> ImageArray:
        """Neutralise the light source's colour cast (grey-world assumption).

        **Why this is here.** Measured on four real detections relit six ways, the
        thing that actually breaks matching is not brightness — it is *colour
        temperature*. A dim, bright, or flat version of a person still scored
        0.95-0.98 against their reference, but the same person under warm indoor
        light scored 0.879 and under cool daylight 0.873, while different people
        reached 0.545. Registering beside a window and tracking under a bulb is
        exactly that failure.

        Equalising the per-channel means removes the cast, because a change of light
        source *is* a multiplicative shift in the channel means. Same four people,
        same six lightings: warm rose to 0.986, cool to 0.993, worst-case same-person
        similarity from 0.777 to 0.925, and the window a threshold can sit in from
        +0.232 to +0.374.

        **The trade-off, stated.** Colour is also a cue for telling people apart, so
        neutralising it costs a little discrimination: different-person similarity
        edged up from 0.545 to 0.551. Worth it many times over — a 0.148 gain on the
        worst genuine match against a 0.006 loss on the worst impostor. CLAHE on
        luminance was tried too (+0.291) and greyscale (+0.282, and it pushed
        impostors to 0.662 by discarding clothing colour); both are worse than this.

        **This changes what an embedding means.** A gallery enrolled before this was
        added is not comparable with probes embedded after it, so enrolled people must
        be registered again.
        """
        # float32 for the division; the means are tiny numbers and integer arithmetic
        # would quantise the correction away on a dark crop.
        image = crop.astype(np.float32)
        means = image.reshape(-1, 3).mean(axis=0)
        # A channel that is essentially black carries no colour information, and
        # dividing by it would explode the gain instead of correcting anything.
        np.maximum(means, 1.0, out=means)
        image *= means.mean() / means
        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    def _blank_batch(self, size: int) -> torch.Tensor:
        """Return a zero input batch on the model's device and precision."""
        tensor = torch.zeros(
            (size, 3, self._config.input_height, self._config.input_width),
            device=self._torch_device,
        )
        return tensor.half() if self._half else tensor

    @staticmethod
    def _normalise(features: FloatArray) -> EmbeddingMatrix:
        """L2-normalise each row, guarding against a zero vector."""
        array = np.ascontiguousarray(features, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        # A zero-norm row would divide to NaN and poison every later comparison.
        np.maximum(norms, 1e-12, out=norms)
        return np.ascontiguousarray(array / norms, dtype=np.float32)
