"""Device resolution and fallback tests.

Driven by a fake torch module rather than the real one so that every branch —
absent torch, absent CUDA, bad index, exhausted VRAM, pre-Volta hardware — is
exercised deterministically on any machine, including CI with no GPU.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mcmot.core.device import DeviceManager
from mcmot.core.errors import DeviceError
from mcmot.core.settings import DeviceSettings

_MIB = 1024 * 1024


class FakeCuda:
    """Minimal stand-in for ``torch.cuda``."""

    def __init__(
        self,
        *,
        available: bool = True,
        count: int = 1,
        name: str = "NVIDIA GeForce RTX 4050 Laptop GPU",
        capability: tuple[int, int] = (8, 9),
        free_mb: int = 5000,
        total_mb: int = 6141,
        mem_info_raises: bool = False,
    ) -> None:
        self._available = available
        self._count = count
        self._name = name
        self._capability = capability
        self._free_mb = free_mb
        self._total_mb = total_mb
        self._mem_info_raises = mem_info_raises
        self.synchronize_calls: list[int | None] = []

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:  # noqa: ARG002
        return self._name

    def get_device_capability(self, index: int) -> tuple[int, int]:  # noqa: ARG002
        return self._capability

    def mem_get_info(self, index: int) -> tuple[int, int]:  # noqa: ARG002
        if self._mem_info_raises:
            raise RuntimeError("driver refused the query")
        return self._free_mb * _MIB, self._total_mb * _MIB

    def get_device_properties(self, index: int) -> Any:  # noqa: ARG002
        return SimpleNamespace(total_memory=self._total_mb * _MIB)

    def synchronize(self, index: int | None = None) -> None:
        self.synchronize_calls.append(index)


def fake_torch(**kwargs: Any) -> Any:
    """Build a fake ``torch`` module exposing only what DeviceManager touches."""
    return SimpleNamespace(
        __version__="2.5.1+cu121",
        version=SimpleNamespace(cuda="12.1"),
        cuda=FakeCuda(**kwargs),
    )


@pytest.fixture
def use_torch(monkeypatch: pytest.MonkeyPatch):
    """Install a fake torch module into DeviceManager."""

    def _install(module: Any) -> None:
        monkeypatch.setattr(DeviceManager, "_import_torch", staticmethod(lambda: module))

    return _install


def manager(**overrides: Any) -> DeviceManager:
    return DeviceManager(DeviceSettings(**overrides))


class TestCudaResolution:
    def test_healthy_gpu_is_selected(self, use_torch) -> None:
        use_torch(fake_torch())

        info = manager().info

        assert info.type == "cuda"
        assert info.index == 0
        assert info.torch_device == "cuda:0"
        assert info.is_cuda
        assert info.fp16 is True
        assert info.total_vram_mb == 6141
        assert info.free_vram_mb == 5000
        assert info.capability == (8, 9)
        assert info.is_fallback is False
        assert info.reason is None

    def test_resolution_is_cached(self, use_torch) -> None:
        """Probing CUDA initialises a context; it must happen exactly once."""
        module = fake_torch()
        use_torch(module)
        calls = 0
        original = module.cuda.is_available

        def counting() -> bool:
            nonlocal calls
            calls += 1
            return original()

        module.cuda.is_available = counting
        device = manager()

        first = device.info
        second = device.info
        third = device.info

        assert calls == 1
        assert first is second is third

    def test_explicit_cpu_preference_skips_probing(self, use_torch) -> None:
        use_torch(fake_torch())

        info = manager(prefer="cpu").info

        assert info.type == "cpu"
        assert info.torch_device == "cpu"
        assert info.is_fallback is False
        assert info.reason is None

    def test_non_zero_device_index(self, use_torch) -> None:
        use_torch(fake_torch(count=2))

        assert manager(index=1).info.torch_device == "cuda:1"


class TestFallback:
    def test_missing_torch_falls_back_to_cpu(self, use_torch) -> None:
        use_torch(None)

        info = manager().info

        assert info.type == "cpu"
        assert info.is_fallback is True
        assert info.reason is not None
        assert "PyTorch" in info.reason

    def test_unavailable_cuda_falls_back_to_cpu(self, use_torch) -> None:
        use_torch(fake_torch(available=False))

        info = manager().info

        assert info.type == "cpu"
        assert info.is_fallback is True
        assert "CUDA is not available" in (info.reason or "")

    def test_out_of_range_index_falls_back_to_cpu(self, use_torch) -> None:
        use_torch(fake_torch(count=1))

        info = manager(index=3).info

        assert info.type == "cpu"
        assert "out of range" in (info.reason or "")

    def test_insufficient_free_vram_falls_back_to_cpu(self, use_torch) -> None:
        """Better to start slow on CPU than to OOM three minutes into a stream."""
        use_torch(fake_torch(free_mb=300))

        info = manager(min_free_vram_mb=1024).info

        assert info.type == "cpu"
        assert "300 MiB" in (info.reason or "")

    def test_vram_check_can_be_disabled(self, use_torch) -> None:
        use_torch(fake_torch(free_mb=300))

        assert manager(min_free_vram_mb=0).info.type == "cuda"

    def test_cpu_never_reports_fp16(self, use_torch) -> None:
        """Half precision on CPU is emulated and slower; config must not force it."""
        use_torch(None)

        assert manager(allow_fp16=True).info.fp16 is False


class TestStrictMode:
    @pytest.mark.parametrize(
        "torch_kwargs",
        [{"available": False}, {"count": 1}],
        ids=["cuda-unavailable", "index-out-of-range"],
    )
    def test_cuda_required_raises_when_unusable(
        self, use_torch, torch_kwargs: dict[str, Any]
    ) -> None:
        use_torch(fake_torch(**torch_kwargs))

        with pytest.raises(DeviceError) as excinfo:
            _ = manager(prefer="cuda", index=5, allow_cpu_fallback=False).info

        assert excinfo.value.code == "device.unavailable"
        assert "reason" in excinfo.value.details

    def test_missing_torch_raises_when_cuda_required(self, use_torch) -> None:
        use_torch(None)

        with pytest.raises(DeviceError):
            _ = manager(prefer="cuda", allow_cpu_fallback=False).info

    def test_auto_preference_never_raises(self, use_torch) -> None:
        """`auto` means "use what is there"; it must not fail the boot."""
        use_torch(None)

        assert manager(prefer="auto", allow_cpu_fallback=False).info.type == "cpu"


class TestCapabilities:
    def test_fp16_disabled_on_pre_volta_hardware(self, use_torch) -> None:
        use_torch(fake_torch(capability=(6, 1), name="GeForce GTX 1050"))

        info = manager(allow_fp16=True).info

        assert info.type == "cuda"
        assert info.fp16 is False
        assert any("FP16 disabled" in w for w in info.warnings)

    def test_fp16_respects_configuration(self, use_torch) -> None:
        use_torch(fake_torch())

        assert manager(allow_fp16=False).info.fp16 is False

    def test_memory_query_failure_degrades_gracefully(self, use_torch) -> None:
        """A driver that refuses mem_get_info must not prevent using the GPU."""
        use_torch(fake_torch(mem_info_raises=True))

        info = manager(min_free_vram_mb=1024).info

        assert info.type == "cuda"
        assert info.total_vram_mb == 6141
        assert info.free_vram_mb is None


class TestRuntime:
    def test_synchronize_is_a_noop_on_cpu(self, use_torch) -> None:
        module = fake_torch()
        use_torch(module)

        manager(prefer="cpu").synchronize()

        assert module.cuda.synchronize_calls == []

    def test_synchronize_targets_the_resolved_device(self, use_torch) -> None:
        module = fake_torch(count=2)
        use_torch(module)

        manager(index=1).synchronize()

        assert module.cuda.synchronize_calls == [1]

    def test_warmup_runs_the_configured_iterations(self, use_torch) -> None:
        use_torch(fake_torch())
        calls = 0

        def step() -> None:
            nonlocal calls
            calls += 1

        durations = manager(warmup_iterations=3).warmup(step)

        assert calls == 3
        assert len(durations) == 3
        assert all(d >= 0.0 for d in durations)

    def test_warmup_can_be_disabled(self, use_torch) -> None:
        use_torch(fake_torch())
        calls = 0

        def step() -> None:
            nonlocal calls
            calls += 1

        assert manager(warmup_iterations=0).warmup(step) == []
        assert calls == 0

    def test_warmup_synchronises_each_iteration(self, use_torch) -> None:
        """Without a sync, warm-up would time queue submission, not compute."""
        module = fake_torch()
        use_torch(module)

        manager(warmup_iterations=2).warmup(lambda: None)

        assert len(module.cuda.synchronize_calls) == 2


class TestSerialisation:
    def test_to_dict_is_api_ready(self, use_torch) -> None:
        use_torch(fake_torch())

        payload = manager().info.to_dict()

        assert payload["type"] == "cuda"
        assert payload["capability"] == "8.9"
        assert payload["torch_version"] == "2.5.1+cu121"
        assert payload["cuda_version"] == "12.1"
        assert payload["warnings"] == []

    def test_to_dict_reports_null_capability_on_cpu(self, use_torch) -> None:
        use_torch(None)

        assert manager().info.to_dict()["capability"] is None
