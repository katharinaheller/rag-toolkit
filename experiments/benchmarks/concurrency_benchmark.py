"""Concurrency scaling benchmark.

Runs a retrieval (or arbitrary callable) workload at several concurrency
levels using a thread pool, then reports per-level wall-clock throughput and
latency percentiles. Useful for showing how a CPU-bound retriever scales
(or fails to scale) with worker count.

The workload callable is supplied by the caller so this benchmark stays
decoupled from the retrieval adapter. On any failure the level simply records
fewer measurements; it never raises.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from experiments.benchmarks.base import (
    BenchmarkOutcome,
    BenchmarkVariant,
    DeviceSpec,
    empty_outcome,
)
from experiments.core.benchmark_stats import (
    make_series,
    parallel_throughput_qps,
    summarise,
)
from experiments.core.gpu_hardware import HardwareMetadata, collect_hardware_metadata
from experiments.core.resource_timeline import ResourceTimeline

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ConcurrencyBenchmark:
    """Benchmark a workload callable across concurrency levels.

    ``workload(payload) -> Any`` is invoked once per task. ``payloads`` are
    cycled to fill ``tasks_per_level`` tasks. The device label is purely
    descriptive (the workload decides its own placement).
    """

    def __init__(
        self,
        name: str,
        workload: Callable[[Any], Any],
        payloads: Sequence[Any],
        concurrency_levels: Sequence[int] = (1, 2, 4, 8),
        tasks_per_level: int = 64,
        warmup_tasks: int = 8,
        device: Optional[DeviceSpec] = None,
        resource_interval_s: float = 0.2,
        timeline_dir: Optional[Path] = None,
    ) -> None:
        self._name = name
        self._workload = workload
        self._payloads = list(payloads)
        self._levels = [int(c) for c in concurrency_levels if int(c) > 0]
        self._tasks = max(1, int(tasks_per_level))
        self._warmup = max(0, int(warmup_tasks))
        self._device = device or DeviceSpec(
            key="cpu", torch_device="cpu", label="CPU", available=True,
        )
        self._resource_interval_s = float(resource_interval_s)
        self._timeline_dir = Path(timeline_dir) if timeline_dir else None

    def run(self) -> List[BenchmarkOutcome]:
        hw = collect_hardware_metadata()
        outcomes: List[BenchmarkOutcome] = []
        if not self._payloads:
            variant = BenchmarkVariant(
                name=f"{self._name}|no-payloads", device=self._device,
            )
            return [empty_outcome(
                "concurrency", variant, "no payloads supplied", hardware=hw,
            )]
        for level in self._levels:
            variant = BenchmarkVariant(
                name=f"{self._name}|c={level}",
                device=self._device,
                parameters={
                    "concurrency": level,
                    "tasks_per_level": self._tasks,
                    "warmup_tasks": self._warmup,
                },
            )
            outcomes.append(self._run_level(variant, level, hw))
        return outcomes

    def _run_level(
        self, variant: BenchmarkVariant, level: int, hw: HardwareMetadata,
    ) -> BenchmarkOutcome:
        started = _utc_iso()
        wall_start = time.perf_counter()
        warnings: List[str] = []

        n_payloads = len(self._payloads)

        # Warmup (serial, ignored).
        for i in range(self._warmup):
            try:
                self._workload(self._payloads[i % n_payloads])
            except Exception as exc:
                warnings.append(f"warmup failed: {exc}")

        latencies_ms: List[float] = []
        timeline = ResourceTimeline(
            interval_s=self._resource_interval_s,
            gpu_monitoring=self._device.torch_device.startswith("cuda"),
        )

        def _task(idx: int) -> float:
            payload = self._payloads[idx % n_payloads]
            t0 = time.perf_counter()
            self._workload(payload)
            return (time.perf_counter() - t0) * 1000.0

        wall_block_start = time.perf_counter()
        with timeline:
            if level == 1:
                for i in range(self._tasks):
                    try:
                        latencies_ms.append(_task(i))
                    except Exception as exc:
                        warnings.append(f"task {i} failed: {exc}")
            else:
                with ThreadPoolExecutor(max_workers=level) as pool:
                    futures = [pool.submit(_task, i) for i in range(self._tasks)]
                    for fut in as_completed(futures):
                        try:
                            latencies_ms.append(fut.result())
                        except Exception as exc:
                            warnings.append(f"task failed: {exc}")
        wall_block_ms = (time.perf_counter() - wall_block_start) * 1000.0

        finished = _utc_iso()
        wall_time = time.perf_counter() - wall_start

        primary = summarise(latencies_ms)
        tput = parallel_throughput_qps(latencies_ms, wall_block_ms)

        timeline_path = self._write_timeline(timeline, variant)

        return BenchmarkOutcome(
            benchmark="concurrency",
            variant=variant,
            success=len(latencies_ms) > 0,
            skipped_reason=None if latencies_ms else "no measurements collected",
            started_iso=started,
            finished_iso=finished,
            wall_time_s=round(wall_time, 4),
            warmup_n=self._warmup,
            measured_n=len(latencies_ms),
            measured_unit="ms",
            series=[make_series("task_latency_ms", latencies_ms, unit="ms")],
            primary_summary=primary,
            throughput_qps=tput,
            resource_timeline_samples=len(timeline.samples),
            resource_timeline_summary=timeline.summary(),
            resource_timeline_path=str(timeline_path) if timeline_path else None,
            hardware=hw,
            metadata={
                "concurrency": level,
                "wall_block_ms": round(wall_block_ms, 3),
                "device": self._device.to_dict(),
            },
            warnings=warnings,
        )

    def _write_timeline(
        self, timeline: ResourceTimeline, variant: BenchmarkVariant,
    ) -> Optional[Path]:
        if self._timeline_dir is None:
            return None
        safe = variant.name.replace("|", "__").replace(":", "-").replace("=", "")
        out_path = self._timeline_dir / f"timeline_{safe}.jsonl"
        try:
            return timeline.write_jsonl(out_path)
        except Exception as exc:
            logger.warning("Failed to write timeline JSONL %s: %s", out_path, exc)
            return None
