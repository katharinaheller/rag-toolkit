"""Descriptive statistics for benchmark measurements.

Pure-stdlib implementations of the metrics every benchmark suite needs:
mean / median / min / max / std-dev / p95 / p99, plus IQR-based outlier
detection. Returns immutable typed records (``BenchmarkSummary``) so the
storage layer can serialise them without further work.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate statistics for a single benchmark measurement series."""

    n: int
    mean: float
    median: float
    min: float
    max: float
    std_dev: float
    p50: float
    p90: float
    p95: float
    p99: float
    iqr: float
    n_outliers: int
    outlier_threshold_low: float
    outlier_threshold_high: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def percentile(values: List[float], pct: float) -> float:
    """Linear interpolation percentile (``pct`` in [0, 100]).

    Matches NumPy's default behaviour on a list of floats. Returns 0.0 on an
    empty list rather than raising.
    """
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(sorted_vals[int(k)])
    frac = k - lo
    return float(sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac)


def _iqr_bounds(values: List[float]) -> Tuple[float, float, float]:
    """Return ``(iqr, low_threshold, high_threshold)`` from Tukey's rule (k=1.5)."""
    if len(values) < 4:
        return 0.0, float("-inf"), float("inf")
    q1 = percentile(values, 25.0)
    q3 = percentile(values, 75.0)
    iqr = q3 - q1
    return iqr, q1 - 1.5 * iqr, q3 + 1.5 * iqr


def summarise(values: List[float]) -> BenchmarkSummary:
    """Compute the full benchmark summary for one series of latencies/durations."""
    n = len(values)
    if n == 0:
        return BenchmarkSummary(
            n=0, mean=0.0, median=0.0, min=0.0, max=0.0, std_dev=0.0,
            p50=0.0, p90=0.0, p95=0.0, p99=0.0,
            iqr=0.0, n_outliers=0,
            outlier_threshold_low=0.0, outlier_threshold_high=0.0,
        )
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std_dev = math.sqrt(variance)

    p50 = percentile(values, 50.0)
    p90 = percentile(values, 90.0)
    p95 = percentile(values, 95.0)
    p99 = percentile(values, 99.0)
    iqr, lo, hi = _iqr_bounds(values)
    n_outliers = sum(1 for v in values if v < lo or v > hi)

    return BenchmarkSummary(
        n=n,
        mean=round(mean, 4),
        median=round(p50, 4),
        min=round(min(values), 4),
        max=round(max(values), 4),
        std_dev=round(std_dev, 4),
        p50=round(p50, 4),
        p90=round(p90, 4),
        p95=round(p95, 4),
        p99=round(p99, 4),
        iqr=round(iqr, 4),
        n_outliers=int(n_outliers),
        outlier_threshold_low=round(lo, 4) if math.isfinite(lo) else 0.0,
        outlier_threshold_high=round(hi, 4) if math.isfinite(hi) else 0.0,
    )


def throughput_qps(latencies_ms: List[float]) -> float:
    """Throughput in queries per second from a list of per-call ms latencies."""
    total = sum(latencies_ms)
    if total <= 0:
        return 0.0
    return round((len(latencies_ms) * 1000.0) / total, 3)


def parallel_throughput_qps(latencies_ms: List[float], wall_time_ms: float) -> float:
    """Wall-clock throughput when calls overlap (concurrency benchmarks)."""
    if wall_time_ms <= 0:
        return 0.0
    return round((len(latencies_ms) * 1000.0) / wall_time_ms, 3)


def speedup(reference_ms: float, candidate_ms: float) -> float:
    """How many times faster ``candidate`` is than ``reference``."""
    if candidate_ms <= 0:
        return 0.0
    return round(reference_ms / candidate_ms, 3)


def items_per_second(n_items: int, elapsed_s: float) -> float:
    if elapsed_s <= 0:
        return 0.0
    return round(n_items / elapsed_s, 3)


@dataclass(frozen=True)
class MeasurementSeries:
    """A labelled series of raw measurements + its summary statistics."""

    label: str
    unit: str
    values: List[float]
    summary: BenchmarkSummary
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "unit": self.unit,
            "n": self.summary.n,
            "summary": self.summary.to_dict(),
            "metadata": self.metadata,
        }


def make_series(
    label: str,
    values: List[float],
    unit: str = "ms",
    metadata: Optional[Dict[str, Any]] = None,
) -> MeasurementSeries:
    """Convenience helper: bundle values + computed summary in one record."""
    return MeasurementSeries(
        label=label, unit=unit, values=list(values),
        summary=summarise(values), metadata=metadata or {},
    )
