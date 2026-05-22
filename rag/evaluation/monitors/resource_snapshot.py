"""Point-in-time CPU/RAM/GPU snapshots. Unavailable values are None.

GPU monitoring is delegated to :class:`GpuMonitor`, which itself is fail-soft
and never imports pynvml at module level.
"""

from __future__ import annotations

import datetime
import os
import socket
from typing import Optional

from rag.evaluation.monitors.gpu_monitor import GpuMonitor
from rag.evaluation.monitors.memory import MemoryMonitor
from rag.evaluation.types import ResourceSnapshot


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ResourceSnapshotCollector:
    """Collects ResourceSnapshot instances with minimal overhead.

    GPU monitoring is opt-in. Set ``gpu_monitoring=True`` to enable it; the
    underlying :class:`GpuMonitor` falls back to "unavailable" if pynvml or a
    driver is missing, so enabling the flag is safe on CPU-only machines.
    """

    def __init__(
        self,
        gpu_monitoring: bool = False,
        gpu_index: int = 0,
        gpu_monitor: Optional[GpuMonitor] = None,
    ) -> None:
        self._gpu_monitoring = gpu_monitoring
        self._memory_monitor = MemoryMonitor()
        self._hostname = socket.gethostname()
        self._pid = os.getpid()

        if gpu_monitoring:
            # Dependency injection keeps the class testable without pynvml.
            self._gpu_monitor: Optional[GpuMonitor] = (
                gpu_monitor if gpu_monitor is not None else GpuMonitor(gpu_index)
            )
        else:
            self._gpu_monitor = None

    def collect(self) -> ResourceSnapshot:
        """Capture a single snapshot. Never raises; unavailable metrics are None."""
        rss = self._memory_monitor.current_rss_mb()
        peak = self._memory_monitor.peak_rss_mb()
        cpu = self._memory_monitor.cpu_percent(interval=0.0)

        gpu_util: Optional[float] = None
        gpu_used: Optional[float] = None
        gpu_total: Optional[float] = None

        if self._gpu_monitor is not None:
            info = self._gpu_monitor.info()
            if info.available:
                gpu_util = info.utilisation_percent
                gpu_used = info.memory_used_mb
                gpu_total = info.memory_total_mb

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
