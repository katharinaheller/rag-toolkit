"""Process-level memory monitor backed by psutil (optional dependency)."""

from __future__ import annotations

import os
from typing import Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class MemoryMonitor:
    """Tracks RSS for the current process. Returns None gracefully when psutil is absent."""

    def __init__(self) -> None:
        self._process = None
        if _PSUTIL_AVAILABLE:
            try:
                self._process = psutil.Process(os.getpid())
            except Exception:
                pass

    @property
    def available(self) -> bool:
        """True when psutil is installed and the process handle is valid."""
        return self._process is not None

    def current_rss_mb(self) -> Optional[float]:
        """Current RSS in megabytes, or None if unavailable."""
        if self._process is None:
            return None
        try:
            mem = self._process.memory_info()
            return mem.rss / (1024 * 1024)
        except Exception:
            return None

    def peak_rss_mb(self) -> Optional[float]:
        """Approximate peak RSS in megabytes, or None if unavailable.

        On Linux/macOS this approximates via current RSS; on Windows psutil
        exposes peak_wset on some versions.
        """
        if self._process is None:
            return None
        try:
            if hasattr(self._process, "memory_info"):
                info = self._process.memory_info()
                return info.rss / (1024 * 1024)
        except Exception:
            pass
        return None

    def cpu_percent(self, interval: float = 0.0) -> Optional[float]:
        """Process CPU utilisation percentage. First non-blocking call returns 0.0."""
        if self._process is None:
            return None
        try:
            return self._process.cpu_percent(interval=interval)
        except Exception:
            return None
