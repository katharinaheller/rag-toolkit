"""Runs a BenchmarkCase repeatedly and computes descriptive statistics.

Warmup iterations are excluded from reported statistics. This matters for local
LLM benchmarks where the first inference call loads the model into VRAM.
"""

from __future__ import annotations

import datetime
import logging
import math
import socket
from typing import Any, Dict, List, Optional

from rag.evaluation.benchmarks.benchmark_case import BenchmarkCase
from rag.evaluation.benchmarks.config import BenchmarkConfig
from rag.evaluation.monitors.memory import MemoryMonitor
from rag.evaluation.types import BenchmarkResult, BenchmarkStats

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _compute_stats(values: List[float]) -> BenchmarkStats:
    """Compute mean, median, min, max, p95, and std_dev over a list of floats."""
    n = len(values)
    if n == 0:
        return BenchmarkStats(
            n=0, mean=0.0, median=0.0, min=0.0, max=0.0, p95=0.0, std_dev=0.0
        )

    sorted_vals = sorted(values)
    mean = sum(sorted_vals) / n
    median = sorted_vals[n // 2] if n % 2 != 0 else (
        (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    )
    minimum = sorted_vals[0]
    maximum = sorted_vals[-1]

    p95_idx = max(0, math.ceil(0.95 * n) - 1)
    p95 = sorted_vals[p95_idx]

    variance = sum((v - mean) ** 2 for v in sorted_vals) / n
    std_dev = math.sqrt(variance)

    return BenchmarkStats(
        n=n,
        mean=round(mean, 3),
        median=round(median, 3),
        min=round(minimum, 3),
        max=round(maximum, 3),
        p95=round(p95, 3),
        std_dev=round(std_dev, 3),
    )


def _collect_hardware_metadata() -> Dict[str, Any]:
    """Collect hardware metadata for benchmark result provenance."""
    meta: Dict[str, Any] = {"hostname": socket.gethostname()}

    mem = MemoryMonitor()
    if mem.available:
        rss = mem.current_rss_mb()
        if rss is not None:
            meta["process_rss_mb_at_start"] = round(rss, 1)

    try:
        import platform
        meta["python_version"] = platform.python_version()
        meta["os"] = platform.system()
        meta["cpu_count"] = _cpu_count()
    except Exception:
        pass

    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024 ** 3)
        meta["total_ram_gb"] = round(total_ram, 1)
    except Exception:
        pass

    return meta


def _cpu_count() -> Optional[int]:
    try:
        import os
        return os.cpu_count()
    except Exception:
        return None


class BenchmarkRunner:
    """Runs a BenchmarkCase with warmup and measured iterations."""

    def __init__(self, config: BenchmarkConfig) -> None:
        self._config = config

    def run(
        self,
        case: BenchmarkCase,
        corpus_size: Optional[int] = None,
        top_k: Optional[int] = None,
        concurrency: Optional[int] = None,
    ) -> BenchmarkResult:
        """Execute the benchmark and return statistics."""
        import time as _time
        timestamp = _utc_iso()
        hw_meta = _collect_hardware_metadata()
        warnings: List[str] = []

        logger.info(
            "BenchmarkRunner | name=%s | warmup=%d | n=%d",
            self._config.name, self._config.warmup_iterations, self._config.n_iterations,
        )

        for i in range(self._config.warmup_iterations):
            try:
                case.run()
            except Exception as exc:
                warnings.append(f"Warmup iteration {i + 1} failed: {exc}")
                logger.warning("Benchmark warmup %d failed: %s", i + 1, exc)

        wall_start = _time.perf_counter()
        measured: List[float] = []
        for i in range(self._config.n_iterations):
            try:
                elapsed_ms = case.run()
                measured.append(elapsed_ms)
            except Exception as exc:
                warnings.append(f"Measured iteration {i + 1} failed: {exc}")
                logger.warning("Benchmark iteration %d failed: %s", i + 1, exc)

        total_s = _time.perf_counter() - wall_start

        if not measured:
            warnings.append(
                "All measured iterations failed. Statistics are meaningless."
            )

        stats = _compute_stats(measured)

        return BenchmarkResult(
            benchmark_name=self._config.name,
            config_dict={
                "n_iterations": self._config.n_iterations,
                "warmup_iterations": self._config.warmup_iterations,
                "top_k": self._config.top_k,
                **self._config.metadata,
                **case.metadata,
            },
            stats=stats,
            raw_values=measured,
            timestamp_iso=timestamp,
            duration_s=round(total_s, 3),
            warmup_n=self._config.warmup_iterations,
            measured_n=len(measured),
            corpus_size=corpus_size,
            top_k=top_k or self._config.top_k,
            concurrency=concurrency,
            model_name=self._config.model_name,
            hardware_metadata=hw_meta,
            warnings=warnings,
            metadata=case.metadata,
        )
