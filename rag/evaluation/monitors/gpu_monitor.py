"""Optional GPU telemetry via pynvml.

The CPU-only build path of the toolkit deliberately does not install pynvml.
GpuMonitor therefore never imports pynvml at module level and never raises:
on a machine without pynvml or without an NVIDIA driver it reports
``available = False`` and returns ``GpuInfo`` instances whose numeric fields
are ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GpuInfo:
    """Snapshot of a single GPU's identity and current utilisation.

    All numeric fields are ``None`` when the metric could not be retrieved.
    """

    available: bool
    name: Optional[str] = None
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    utilisation_percent: Optional[float] = None
    error: Optional[str] = None

    @classmethod
    def unavailable(cls, reason: str = "") -> "GpuInfo":
        """Return a sentinel value used when no GPU telemetry is reachable."""
        return cls(available=False, error=reason or None)

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "name": self.name,
            "memory_total_mb": self.memory_total_mb,
            "memory_used_mb": self.memory_used_mb,
            "utilisation_percent": self.utilisation_percent,
            "error": self.error,
        }


class GpuMonitor:
    """Reads GPU telemetry via pynvml when available.

    The monitor is intentionally fail-soft: missing pynvml, missing driver,
    or any pynvml call failure resolves to ``available = False`` and an
    ``unavailable`` GpuInfo. This keeps the CPU-only deployment path free of
    a hard pynvml dependency.

    A single instance is bound to a fixed GPU index (default 0). Create a
    new instance to query a different device.
    """

    def __init__(self, gpu_index: int = 0) -> None:
        self._gpu_index = gpu_index
        self._pynvml: Optional[Any] = None
        self._handle: Optional[Any] = None
        self._init_error: Optional[str] = None
        self._init()

    # ── lifecycle ────────────────────────────────────────────────────────

    def _init(self) -> None:
        """Try to import pynvml and bind a device handle. Never raises."""
        try:
            import pynvml  # type: ignore[import-not-found]
        except ImportError as exc:
            self._init_error = f"pynvml not installed: {exc}"
            return

        try:
            pynvml.nvmlInit()
        except Exception as exc:  # pynvml raises NVMLError, plus driver issues.
            self._init_error = f"nvmlInit failed: {exc}"
            return

        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            self._pynvml = pynvml
        except Exception as exc:
            self._init_error = f"device index {self._gpu_index} not reachable: {exc}"
            self._handle = None
            self._pynvml = None

    # ── public API ───────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True only when pynvml is importable AND a device handle is bound."""
        return self._handle is not None and self._pynvml is not None

    def info(self) -> GpuInfo:
        """Return current GPU telemetry, or ``GpuInfo.unavailable()``."""
        if not self.available:
            return GpuInfo.unavailable(self._init_error or "GPU not available")

        try:
            name = self._pynvml.nvmlDeviceGetName(self._handle)  # type: ignore[union-attr]
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")

            mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)  # type: ignore[union-attr]
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)  # type: ignore[union-attr]

            return GpuInfo(
                available=True,
                name=name,
                memory_total_mb=float(mem.total) / (1024 * 1024),
                memory_used_mb=float(mem.used) / (1024 * 1024),
                utilisation_percent=float(util.gpu),
            )
        except Exception as exc:
            return GpuInfo.unavailable(f"pynvml query failed: {exc}")

    def shutdown(self) -> None:
        """Release the NVML library handle. Safe to call multiple times."""
        if self._pynvml is None:
            return
        try:
            self._pynvml.nvmlShutdown()
        except Exception:
            pass
        finally:
            self._pynvml = None
            self._handle = None
