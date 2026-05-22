"""Tests for GpuMonitor and its integration with ResourceSnapshotCollector.

These tests assume **no GPU is available** on the test machine. They verify
that the monitor returns clean fallbacks instead of raising.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any
from unittest import mock

import pytest

from rag.evaluation.monitors.gpu_monitor import GpuInfo, GpuMonitor
from rag.evaluation.monitors.resource_snapshot import ResourceSnapshotCollector


class TestGpuInfo:
    def test_unavailable_factory_marks_available_false(self):
        info = GpuInfo.unavailable()
        assert info.available is False
        assert info.name is None
        assert info.memory_total_mb is None
        assert info.memory_used_mb is None
        assert info.utilisation_percent is None

    def test_unavailable_factory_records_reason(self):
        info = GpuInfo.unavailable("driver missing")
        assert info.error == "driver missing"

    def test_unavailable_factory_empty_reason_yields_none(self):
        info = GpuInfo.unavailable("")
        assert info.error is None

    def test_to_dict_contains_all_fields(self):
        info = GpuInfo(
            available=True,
            name="Test GPU",
            memory_total_mb=8192.0,
            memory_used_mb=512.0,
            utilisation_percent=42.0,
        )
        d = info.to_dict()
        assert d["available"] is True
        assert d["name"] == "Test GPU"
        assert d["memory_total_mb"] == 8192.0
        assert d["memory_used_mb"] == 512.0
        assert d["utilisation_percent"] == 42.0
        assert d["error"] is None


class TestGpuMonitorWithoutPynvml:
    """No pynvml installed (the default in the CPU-only deployment)."""

    def test_monitor_unavailable_when_pynvml_missing(self, monkeypatch):
        # Simulate ImportError on `import pynvml`.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pynvml":
                raise ImportError("simulated: pynvml not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            monitor = GpuMonitor()
            assert monitor.available is False
            info = monitor.info()
            assert info.available is False
            assert info.error is not None
            assert "pynvml not installed" in info.error

    def test_info_returns_none_metrics_when_unavailable(self, monkeypatch):
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pynvml":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            info = GpuMonitor().info()
            assert info.memory_total_mb is None
            assert info.memory_used_mb is None
            assert info.utilisation_percent is None
            assert info.name is None


class TestGpuMonitorWithMockedPynvml:
    """Pynvml present but driver / device handle absent or failing."""

    def _install_fake_pynvml(self, monkeypatch, *, init_raises=False,
                             handle_raises=False, query_raises=False):
        fake = types.ModuleType("pynvml")

        def nvmlInit() -> None:
            if init_raises:
                raise RuntimeError("nvmlInit failed")

        def nvmlShutdown() -> None:
            return None

        def nvmlDeviceGetHandleByIndex(i: int) -> object:
            if handle_raises:
                raise RuntimeError(f"no device at index {i}")
            return object()  # opaque handle.

        class _Mem:
            total = 8 * 1024 * 1024 * 1024
            used = 1 * 1024 * 1024 * 1024

        class _Util:
            gpu = 17

        def nvmlDeviceGetMemoryInfo(h: object) -> _Mem:
            if query_raises:
                raise RuntimeError("query failed")
            return _Mem()

        def nvmlDeviceGetUtilizationRates(h: object) -> _Util:
            if query_raises:
                raise RuntimeError("query failed")
            return _Util()

        def nvmlDeviceGetName(h: object) -> str:
            if query_raises:
                raise RuntimeError("query failed")
            return "Fake GPU 9999"

        fake.nvmlInit = nvmlInit
        fake.nvmlShutdown = nvmlShutdown
        fake.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
        fake.nvmlDeviceGetMemoryInfo = nvmlDeviceGetMemoryInfo
        fake.nvmlDeviceGetUtilizationRates = nvmlDeviceGetUtilizationRates
        fake.nvmlDeviceGetName = nvmlDeviceGetName

        monkeypatch.setitem(sys.modules, "pynvml", fake)
        return fake

    def test_init_failure_yields_unavailable(self, monkeypatch):
        self._install_fake_pynvml(monkeypatch, init_raises=True)
        monitor = GpuMonitor()
        assert monitor.available is False
        info = monitor.info()
        assert info.available is False
        assert "nvmlInit failed" in (info.error or "")

    def test_handle_failure_yields_unavailable(self, monkeypatch):
        self._install_fake_pynvml(monkeypatch, handle_raises=True)
        monitor = GpuMonitor(gpu_index=7)
        assert monitor.available is False
        assert "index 7" in (monitor.info().error or "")

    def test_query_failure_yields_unavailable_info(self, monkeypatch):
        self._install_fake_pynvml(monkeypatch, query_raises=True)
        monitor = GpuMonitor()
        # Handle binds OK because that path uses different methods.
        assert monitor.available is True
        info = monitor.info()
        assert info.available is False
        assert "query failed" in (info.error or "")

    def test_happy_path_returns_metrics(self, monkeypatch):
        self._install_fake_pynvml(monkeypatch)
        monitor = GpuMonitor()
        assert monitor.available is True
        info = monitor.info()
        assert info.available is True
        assert info.name == "Fake GPU 9999"
        assert info.memory_total_mb == pytest.approx(8192.0, rel=1e-6)
        assert info.memory_used_mb == pytest.approx(1024.0, rel=1e-6)
        assert info.utilisation_percent == pytest.approx(17.0)

    def test_bytes_name_is_decoded(self, monkeypatch):
        fake = self._install_fake_pynvml(monkeypatch)

        def nvmlDeviceGetName(h: object) -> bytes:
            return b"Bytes GPU"

        fake.nvmlDeviceGetName = nvmlDeviceGetName
        info = GpuMonitor().info()
        assert info.name == "Bytes GPU"

    def test_shutdown_is_safe_when_unavailable(self, monkeypatch):
        # No fake pynvml; monitor sits unavailable.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pynvml":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            monitor = GpuMonitor()
            monitor.shutdown()  # must not raise
            monitor.shutdown()  # idempotent


class TestResourceSnapshotCollectorWithoutGpu:
    """Default collector path on CPU-only deployments."""

    def test_collector_constructed_without_gpu_flag(self):
        collector = ResourceSnapshotCollector()
        snap = collector.collect()
        assert snap.gpu_utilisation_percent is None
        assert snap.gpu_memory_used_mb is None
        assert snap.gpu_memory_total_mb is None

    def test_collector_with_gpu_monitoring_safe_without_pynvml(self, monkeypatch):
        # Even if the caller opts in, no pynvml means we get None metrics —
        # not an exception.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) \
            else __import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pynvml":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            collector = ResourceSnapshotCollector(gpu_monitoring=True)
            snap = collector.collect()
            assert snap.gpu_utilisation_percent is None
            assert snap.gpu_memory_used_mb is None
            assert snap.gpu_memory_total_mb is None

    def test_collector_injected_monitor_overrides_default(self):
        # Inject a stub monitor that always reports available.
        class StubMonitor:
            available = True

            def info(self):
                return GpuInfo(
                    available=True,
                    name="Stub",
                    memory_total_mb=1000.0,
                    memory_used_mb=250.0,
                    utilisation_percent=33.0,
                )

        collector = ResourceSnapshotCollector(
            gpu_monitoring=True,
            gpu_monitor=StubMonitor(),  # type: ignore[arg-type]
        )
        snap = collector.collect()
        assert snap.gpu_memory_total_mb == 1000.0
        assert snap.gpu_memory_used_mb == 250.0
        assert snap.gpu_utilisation_percent == 33.0


class TestEvaluationConfigGpuDefault:
    """gpu_monitoring must remain False by default — CPU-only is the standard."""

    def test_gpu_monitoring_disabled_by_default(self):
        from rag.evaluation.config import EvaluationConfig
        config = EvaluationConfig()
        assert config.gpu_monitoring is False
        assert config.capture_resources is False
