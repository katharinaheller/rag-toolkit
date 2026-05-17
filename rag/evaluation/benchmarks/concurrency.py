"""Concurrency benchmark via a thread pool.

Threads provide real concurrency for network-bound Ollama calls (the GIL is
released during I/O). For CPU-bound work the GIL limits scaling.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from rag.evaluation.benchmarks.config import BenchmarkConfig, ScalingConfig
from rag.evaluation.benchmarks.benchmark_runner import _compute_stats, _collect_hardware_metadata, _utc_iso
from rag.evaluation.types import BenchmarkResult

logger = logging.getLogger(__name__)


class ConcurrencyExperiment:
    """Simulates concurrent users by running a callable from a thread pool."""

    def __init__(
        self,
        base_config: BenchmarkConfig,
        callable_fn: Callable[[], None],
        scaling_config: Optional[ScalingConfig] = None,
    ) -> None:
        self._base_config = base_config
        self._fn = callable_fn
        self._scaling_config = scaling_config or ScalingConfig()

    def run(self) -> List[BenchmarkResult]:
        """Return one BenchmarkResult per concurrency level."""
        results: List[BenchmarkResult] = []

        for level in self._scaling_config.concurrency_levels:
            logger.info(
                "ConcurrencyExperiment | concurrency=%d | n=%d",
                level, self._base_config.n_iterations,
            )
            results.append(self._run_at_level(level))

        return results

    def _run_at_level(self, concurrency: int) -> BenchmarkResult:
        n_total = self._base_config.n_iterations
        warmup = self._base_config.warmup_iterations
        warnings: List[str] = []
        hw_meta = _collect_hardware_metadata()
        timestamp = _utc_iso()

        # Single-threaded warmup avoids race conditions during model loading.
        for _ in range(warmup):
            try:
                self._fn()
            except Exception:
                pass

        start_wall = time.perf_counter()
        measured: List[float] = []
        errors = 0

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(self._timed_call) for _ in range(n_total)]
            for future in as_completed(futures):
                try:
                    elapsed_ms = future.result()
                    measured.append(elapsed_ms)
                except Exception as exc:
                    errors += 1
                    warnings.append(f"Request failed: {exc}")

        total_s = time.perf_counter() - start_wall

        if errors > 0:
            warnings.append(
                f"{errors}/{n_total} requests failed at concurrency={concurrency}."
            )

        stats = _compute_stats(measured)

        return BenchmarkResult(
            benchmark_name=f"{self._base_config.name}_c{concurrency}",
            config_dict={
                "n_iterations": n_total,
                "warmup_iterations": warmup,
                "concurrency": concurrency,
            },
            stats=stats,
            raw_values=measured,
            timestamp_iso=timestamp,
            duration_s=round(total_s, 3),
            warmup_n=warmup,
            measured_n=len(measured),
            concurrency=concurrency,
            model_name=self._base_config.model_name,
            hardware_metadata=hw_meta,
            warnings=warnings,
        )

    def _timed_call(self) -> float:
        t0 = time.perf_counter()
        self._fn()
        return (time.perf_counter() - t0) * 1_000.0
