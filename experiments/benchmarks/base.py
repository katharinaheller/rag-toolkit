"""Shared dataclasses and helpers for benchmark runners.

A :class:`BenchmarkOutcome` is everything we want to keep after a benchmark
variant finishes: the raw measurement series, the computed statistics, the
resource timeline samples, the hardware metadata, and a thin metadata bag for
the run-specific bookkeeping (corpus, batch size, etc).

Outcomes are designed to round-trip through JSON cleanly.
"""

from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from experiments.core.benchmark_stats import BenchmarkSummary, MeasurementSeries
from experiments.core.gpu_hardware import HardwareMetadata
from experiments.core.resource_timeline import ResourceSample


def _utc_iso() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True)
class DeviceSpec:
    """Identifies a benchmark target device.

    ``key`` is the short label used in tables / figures ("cpu", "cuda:0").
    ``available`` is set during construction; benchmarks read it to decide
    whether to skip the variant.
    """

    key: str
    torch_device: str
    label: str
    available: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def base_devices(prefer_gpu_index: int = 0) -> List[DeviceSpec]:
    """Return the standard CPU + GPU device specs for benchmark sweeps.

    GPU is included with ``available=False`` when CUDA is not present, so the
    plotting / reporting layer can still show a "skipped" row.
    """
    cpu = DeviceSpec(
        key="cpu", torch_device="cpu", label="CPU",
        available=True, reason="",
    )

    try:
        import torch  # type: ignore
        cuda_ok = bool(torch.cuda.is_available())
        gpu_name = None
        if cuda_ok:
            try:
                gpu_name = torch.cuda.get_device_name(prefer_gpu_index)
            except Exception:
                gpu_name = None
        gpu = DeviceSpec(
            key=f"cuda:{prefer_gpu_index}",
            torch_device=f"cuda:{prefer_gpu_index}",
            label=f"GPU ({gpu_name})" if gpu_name else "GPU",
            available=cuda_ok,
            reason="" if cuda_ok else "torch.cuda.is_available() = False",
        )
    except Exception as exc:
        gpu = DeviceSpec(
            key=f"cuda:{prefer_gpu_index}",
            torch_device=f"cuda:{prefer_gpu_index}",
            label="GPU",
            available=False,
            reason=f"torch import failed: {exc}",
        )

    return [cpu, gpu]


@dataclass
class BenchmarkVariant:
    """Description of one cell in a benchmark sweep (CPU vs GPU, batch=N, …)."""

    name: str
    device: DeviceSpec
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device.to_dict(),
            "parameters": self.parameters,
        }


@dataclass
class BenchmarkOutcome:
    """The full result of executing one benchmark variant.

    The dataclass is intentionally flat so a CSV exporter can write one row
    per outcome by reading the ``summary`` dict and the ``metadata`` dict.
    """

    benchmark: str
    variant: BenchmarkVariant
    success: bool
    skipped_reason: Optional[str]
    started_iso: str
    finished_iso: str
    wall_time_s: float

    warmup_n: int
    measured_n: int
    measured_unit: str

    series: List[MeasurementSeries] = field(default_factory=list)
    primary_summary: Optional[BenchmarkSummary] = None
    throughput_qps: Optional[float] = None
    items_per_second: Optional[float] = None

    cold_start_ms: Optional[float] = None
    warm_start_ms: Optional[float] = None

    gpu_peak_memory_mb: Optional[float] = None
    gpu_allocated_memory_mb: Optional[float] = None

    resource_timeline_samples: int = 0
    resource_timeline_summary: Dict[str, Any] = field(default_factory=dict)
    resource_timeline_path: Optional[str] = None

    hardware: Optional[HardwareMetadata] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "benchmark": self.benchmark,
            "variant": self.variant.to_dict(),
            "success": self.success,
            "skipped_reason": self.skipped_reason,
            "started_iso": self.started_iso,
            "finished_iso": self.finished_iso,
            "wall_time_s": self.wall_time_s,
            "warmup_n": self.warmup_n,
            "measured_n": self.measured_n,
            "measured_unit": self.measured_unit,
            "series": [s.to_dict() for s in self.series],
            "primary_summary": (
                self.primary_summary.to_dict() if self.primary_summary else None
            ),
            "throughput_qps": self.throughput_qps,
            "items_per_second": self.items_per_second,
            "cold_start_ms": self.cold_start_ms,
            "warm_start_ms": self.warm_start_ms,
            "gpu_peak_memory_mb": self.gpu_peak_memory_mb,
            "gpu_allocated_memory_mb": self.gpu_allocated_memory_mb,
            "resource_timeline_samples": self.resource_timeline_samples,
            "resource_timeline_summary": self.resource_timeline_summary,
            "resource_timeline_path": self.resource_timeline_path,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }
        return out

    def flat_row(self) -> Dict[str, Any]:
        """Project the outcome into a single CSV-friendly row.

        Picks the most important fields (device, parameters, summary stats)
        and flattens nested dicts. Raw measurement series are persisted via
        ``BenchmarkOutcome.to_dict()`` to JSONL — they don't belong in CSV.
        """
        row: Dict[str, Any] = {
            "benchmark": self.benchmark,
            "variant": self.variant.name,
            "device": self.variant.device.key,
            "device_label": self.variant.device.label,
            "device_available": self.variant.device.available,
            "success": self.success,
            "skipped_reason": self.skipped_reason,
            "wall_time_s": self.wall_time_s,
            "warmup_n": self.warmup_n,
            "measured_n": self.measured_n,
            "measured_unit": self.measured_unit,
            "throughput_qps": self.throughput_qps,
            "items_per_second": self.items_per_second,
            "cold_start_ms": self.cold_start_ms,
            "warm_start_ms": self.warm_start_ms,
            "gpu_peak_memory_mb": self.gpu_peak_memory_mb,
            "gpu_allocated_memory_mb": self.gpu_allocated_memory_mb,
        }
        for k, v in self.variant.parameters.items():
            row[f"param.{k}"] = v
        if self.primary_summary is not None:
            for k, v in self.primary_summary.to_dict().items():
                row[f"stats.{k}"] = v
        tl = self.resource_timeline_summary or {}
        for k in ("cpu_percent", "memory_rss_mb",
                  "gpu_utilisation_percent", "gpu_memory_used_mb",
                  "gpu_torch_allocated_mb", "gpu_torch_peak_mb"):
            inner = tl.get(k)
            if isinstance(inner, dict):
                row[f"timeline.{k}.mean"] = inner.get("mean")
                row[f"timeline.{k}.max"] = inner.get("max")
        return row


def empty_outcome(
    benchmark: str,
    variant: BenchmarkVariant,
    reason: str,
    hardware: Optional[HardwareMetadata] = None,
) -> BenchmarkOutcome:
    """Build an outcome representing a deliberately skipped variant."""
    now = _utc_iso()
    return BenchmarkOutcome(
        benchmark=benchmark,
        variant=variant,
        success=False,
        skipped_reason=reason,
        started_iso=now,
        finished_iso=now,
        wall_time_s=0.0,
        warmup_n=0,
        measured_n=0,
        measured_unit="ms",
        hardware=hardware,
    )
