"""Collects point-in-time CPU/RAM/GPU snapshots. Unavailable values are None.

GPU monitoring uses pynvml when available.
"""

from __future__ import annotations

import datetime
import os
import socket
from typing import Optional

from rag.evaluation.monitors.memory import MemoryMonitor
from rag.evaluation.types import ResourceSnapshot

try:
    import pynvml
    _PYNVML_AVAILABLE = True
    try:
        pynvml.nvmlInit()
        _NVML_INITIALISED = True
    except Exception:
        _NVML_INITIALISED = False
except ImportError:
    _PYNVML_AVAILABLE = False
    _NVML_INITIALISED = False


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _get_gpu_stats(gpu_index: int = 0):
    """Return (utilisation_pct, used_mb, total_mb) or (None, None, None)."""
    if not _NVML_INITIALISED:
        return None, None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (
            float(util.gpu),
            float(mem.used) / (1024 * 1024),
            float(mem.total) / (1024 * 1024),
        )
    except Exception:
        return None, None, None


class ResourceSnapshotCollector:
    """Collects ResourceSnapshot instances with minimal overhead."""

    def __init__(self, gpu_monitoring: bool = False, gpu_index: int = 0) -> None:
        self._gpu_monitoring = gpu_monitoring
        self._gpu_index = gpu_index
        self._memory_monitor = MemoryMonitor()
        self._hostname = socket.gethostname()
        self._pid = os.getpid()

    def collect(self) -> ResourceSnapshot:
        """Capture a single snapshot. Never raises; unavailable metrics are None."""
        rss = self._memory_monitor.current_rss_mb()
        peak = self._memory_monitor.peak_rss_mb()
        cpu = self._memory_monitor.cpu_percent(interval=0.0)

        gpu_util: Optional[float] = None
        gpu_used: Optional[float] = None
        gpu_total: Optional[float] = None

        if self._gpu_monitoring:
            gpu_util, gpu_used, gpu_total = _get_gpu_stats(self._gpu_index)

        return ResourceSnapshot(
            timestamp_iso=_utc_iso(),
            cpu_percent=cpu,
            memory_rss_mb=rss,
            memory_peak_mb=peak,
            gpu_utilisation_percent=gpu_util,
            gpu_memory_used_mb=gpu_used,
            gpu_memory_total_mb=gpu_total,
            process_id=self._pid,
            hostname=self._hostname,
        )
