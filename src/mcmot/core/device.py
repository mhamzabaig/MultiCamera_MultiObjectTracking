"""Compute device selection, capability probing, and warm-up.

Every model in the system asks this module where to run instead of calling
``torch.cuda.is_available()`` itself. That centralisation buys three things:

* **One fallback policy.** A GPU that is absent, out of range, or nearly full is
  handled identically everywhere, and the decision is logged once with a reason.
* **No hard torch dependency at import time.** Configuration, logging, the API
  layer, and their tests all import this module on a machine with no torch and
  no CUDA. Torch is imported lazily, and its absence degrades to CPU rather than
  crashing the process.
* **Honest FP16.** Half precision is enabled only when the hardware actually
  benefits from it, so ``half: true`` in config cannot silently make a CPU or an
  old GPU *slower*.

Fallback is deliberate policy, not an accident: for a surveillance workload a
degraded system beats a dead one, so an unusable GPU produces a warning and a
CPU device. Set ``device.allow_cpu_fallback: false`` to make it fatal instead —
appropriate for a benchmarking run, where silently falling back to CPU would
invalidate the numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from mcmot.core.errors import DeviceError
from mcmot.core.logging import get_logger
from mcmot.core.settings import DeviceSettings

__all__ = ["DeviceInfo", "DeviceManager"]

_log = get_logger(__name__)

T = TypeVar("T")

#: Compute capability at which FP16 tensor cores become a real win (Volta+).
_MIN_FP16_CAPABILITY = (7, 0)
_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    """An immutable description of the resolved compute device."""

    type: str
    """Either ``"cuda"`` or ``"cpu"``."""

    index: int | None
    """CUDA ordinal, or ``None`` on CPU."""

    name: str
    """Human-readable device name."""

    fp16: bool
    """Whether half precision should actually be used."""

    total_vram_mb: int | None = None
    free_vram_mb: int | None = None
    capability: tuple[int, int] | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    is_fallback: bool = False
    """True when the resolved device is not the one that was requested."""

    reason: str | None = None
    """Why a fallback happened. ``None`` on a clean resolution."""

    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def torch_device(self) -> str:
        """Device string accepted by torch and Ultralytics (``"cuda:0"``/``"cpu"``)."""
        if self.type == "cuda" and self.index is not None:
            return f"cuda:{self.index}"
        return self.type

    @property
    def is_cuda(self) -> bool:
        """Whether the device is a CUDA GPU."""
        return self.type == "cuda"

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable view for ``/api/system/health`` and boot logs."""
        return {
            "type": self.type,
            "index": self.index,
            "name": self.name,
            "fp16": self.fp16,
            "total_vram_mb": self.total_vram_mb,
            "free_vram_mb": self.free_vram_mb,
            "capability": (
                f"{self.capability[0]}.{self.capability[1]}" if self.capability else None
            ),
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "is_fallback": self.is_fallback,
            "reason": self.reason,
            "warnings": list(self.warnings),
        }


class DeviceManager:
    """Resolves and caches the compute device for the process lifetime.

    Resolution runs once — probing CUDA costs hundreds of milliseconds on the
    first call because it initialises the CUDA context, so doing it per model or
    per frame would be wasteful and would let the answer change mid-run.
    """

    def __init__(self, config: DeviceSettings) -> None:
        """Args: config: Validated device configuration."""
        self._config = config
        self._info: DeviceInfo | None = None

    # -- resolution -------------------------------------------------------
    @property
    def info(self) -> DeviceInfo:
        """Return the resolved device, probing hardware on first access."""
        if self._info is None:
            self._info = self._resolve()
            self._log_resolution(self._info)
        return self._info

    def _resolve(self) -> DeviceInfo:
        prefer = self._config.prefer
        torch = self._import_torch()

        if torch is None:
            if prefer == "cpu":
                return self._cpu_info()
            # Under "auto" a GPU was still preferred, so landing on CPU is a
            # reported degradation rather than a clean resolution.
            return self._cpu_or_fail("PyTorch is not installed", fatal=prefer == "cuda")

        torch_version = str(torch.__version__)
        cuda_version = getattr(torch.version, "cuda", None)

        if prefer == "cpu":
            return self._cpu_info(torch_version=torch_version, cuda_version=cuda_version)

        if not torch.cuda.is_available():
            return self._cpu_or_fail(
                "CUDA is not available to PyTorch "
                f"(torch {torch_version}, cuda build {cuda_version or 'none'})",
                torch_version=torch_version,
                cuda_version=cuda_version,
                fatal=prefer == "cuda",
            )

        count = int(torch.cuda.device_count())
        index = self._config.index
        if index >= count:
            return self._cpu_or_fail(
                f"CUDA device index {index} out of range ({count} device(s) present)",
                torch_version=torch_version,
                cuda_version=cuda_version,
                fatal=prefer == "cuda",
            )

        name = str(torch.cuda.get_device_name(index))
        capability = tuple(torch.cuda.get_device_capability(index))
        total_mb, free_mb = self._query_memory(torch, index)

        warnings: list[str] = []
        if (
            free_mb is not None
            and self._config.min_free_vram_mb
            and free_mb < self._config.min_free_vram_mb
        ):
            return self._cpu_or_fail(
                f"only {free_mb} MiB of VRAM free, "
                f"{self._config.min_free_vram_mb} MiB required",
                torch_version=torch_version,
                cuda_version=cuda_version,
                fatal=prefer == "cuda",
            )

        fp16 = self._config.allow_fp16
        if fp16 and capability < _MIN_FP16_CAPABILITY:
            fp16 = False
            warnings.append(
                f"FP16 disabled: compute capability {capability[0]}.{capability[1]} "
                "predates efficient half precision"
            )

        return DeviceInfo(
            type="cuda",
            index=index,
            name=name,
            fp16=fp16,
            total_vram_mb=total_mb,
            free_vram_mb=free_mb,
            capability=(capability[0], capability[1]),
            torch_version=torch_version,
            cuda_version=cuda_version,
            warnings=tuple(warnings),
        )

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _import_torch() -> Any | None:
        try:
            import torch  # noqa: PLC0415 - intentionally lazy
        except ImportError:
            return None
        return torch

    @staticmethod
    def _query_memory(torch: Any, index: int) -> tuple[int | None, int | None]:
        """Return ``(total_mb, free_mb)``, or ``(None, None)`` if unavailable."""
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:  # noqa: BLE001 - driver quirks must not break startup
            try:
                props = torch.cuda.get_device_properties(index)
            except Exception:  # noqa: BLE001
                return None, None
            return int(props.total_memory) // _BYTES_PER_MIB, None
        return int(total_bytes) // _BYTES_PER_MIB, int(free_bytes) // _BYTES_PER_MIB

    def _cpu_info(
        self,
        *,
        reason: str | None = None,
        is_fallback: bool = False,
        torch_version: str | None = None,
        cuda_version: str | None = None,
    ) -> DeviceInfo:
        return DeviceInfo(
            type="cpu",
            index=None,
            name="cpu",
            # FP16 on CPU is emulated and slower than FP32 for our models.
            fp16=False,
            torch_version=torch_version,
            cuda_version=cuda_version,
            is_fallback=is_fallback,
            reason=reason,
        )

    def _cpu_or_fail(
        self,
        reason: str,
        *,
        torch_version: str | None = None,
        cuda_version: str | None = None,
        fatal: bool = True,
    ) -> DeviceInfo:
        """Fall back to CPU, or raise when the operator forbade fallback."""
        if fatal and not self._config.allow_cpu_fallback:
            raise DeviceError(
                f"CUDA was requested but is unusable: {reason}",
                details={
                    "prefer": self._config.prefer,
                    "index": self._config.index,
                    "reason": reason,
                },
            )
        return self._cpu_info(
            reason=reason,
            is_fallback=True,
            torch_version=torch_version,
            cuda_version=cuda_version,
        )

    @staticmethod
    def _log_resolution(info: DeviceInfo) -> None:
        payload = info.to_dict()
        if info.is_fallback:
            _log.warning("device.fallback", **payload)
        else:
            _log.info("device.resolved", **payload)
        for message in info.warnings:
            _log.warning("device.capability_warning", detail=message)

    # -- runtime ----------------------------------------------------------
    def synchronize(self) -> None:
        """Block until queued GPU work completes. A no-op on CPU.

        Required before any latency measurement: CUDA kernel launches are
        asynchronous, so timing without this measures queueing, not compute.
        """
        if not self.info.is_cuda:
            return
        torch = self._import_torch()
        if torch is not None:
            torch.cuda.synchronize(self.info.index)

    def warmup(
        self,
        step: Callable[[], T],
        *,
        iterations: int | None = None,
        label: str = "model",
    ) -> list[float]:
        """Run ``step`` a few times to absorb one-off initialisation costs.

        The first CUDA forward pass of a process pays for context creation,
        kernel autotuning, and cuDNN algorithm selection — commonly 1–2 seconds.
        Paying it at startup keeps it out of the first tracked frame, where it
        would show up as a stall and a dropped stream.

        Args:
            step: Zero-argument callable performing one representative inference.
            iterations: Override the configured iteration count.
            label: Name used in the emitted log line.

        Returns:
            Per-iteration wall-clock durations in seconds.
        """
        count = self._config.warmup_iterations if iterations is None else iterations
        if count <= 0:
            return []

        durations: list[float] = []
        for _ in range(count):
            start = time.perf_counter()
            step()
            self.synchronize()
            durations.append(time.perf_counter() - start)

        _log.info(
            "device.warmup_complete",
            label=label,
            iterations=count,
            first_ms=round(durations[0] * 1000, 2),
            last_ms=round(durations[-1] * 1000, 2),
            device=self.info.torch_device,
        )
        return durations
