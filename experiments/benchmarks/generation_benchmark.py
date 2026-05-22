"""Generation benchmark.

Generation is currently backed by Ollama (an external server). On the rag-toolkit
stack we cannot directly toggle CPU vs GPU for an Ollama call from Python, so
the benchmark records *what the server reports*: per-call wall latency,
token throughput, and the resource timeline of the *local* process. When the
server runs on the same host as the experiment the local GPU monitor still
captures the VRAM curve of the inference process.

If generation is disabled in settings, the benchmark resolves to a single
"skipped" outcome. The interface mirrors :class:`EmbeddingBenchmark` so the
suite layer can treat both uniformly.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from experiments.benchmarks.base import (
    BenchmarkOutcome,
    BenchmarkVariant,
    DeviceSpec,
    base_devices,
    empty_outcome,
)
from experiments.core.benchmark_stats import (
    items_per_second,
    make_series,
    summarise,
    throughput_qps,
)
from experiments.core.gpu_hardware import (
    HardwareMetadata,
    collect_hardware_metadata,
    cuda_is_available,
    gpu_peak_memory_mb,
    reset_gpu_peak_memory,
)
from experiments.core.resource_timeline import ResourceTimeline

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class GenerationBenchmark:
    """Benchmark generation calls across pseudo-devices (cpu/gpu views).

    Each device variant runs the same generator (Ollama) and records the
    *local* resource timeline. The "device" label here is descriptive — it
    documents which device the Ollama backend was configured to use, but
    actual placement is server-side. This is still useful because the local
    monitor still observes upstream GPU usage when the server is local.
    """

    def __init__(
        self,
        generator,                             # BuiltGenerator
        queries: Sequence[str],
        contexts_per_query: Sequence[List[str]],
        warmup_iterations: int = 1,
        measured_iterations: int = 5,
        devices: Optional[Sequence[DeviceSpec]] = None,
        resource_interval_s: float = 0.25,
        timeline_dir: Optional[Path] = None,
    ) -> None:
        if len(queries) != len(contexts_per_query):
            raise ValueError(
                "queries and contexts_per_query must have the same length."
            )
        self._generator = generator
        self._queries = list(queries)
        self._contexts = [list(c) for c in contexts_per_query]
        self._warmup = max(0, int(warmup_iterations))
        self._measured = max(1, int(measured_iterations))
        self._devices = list(devices) if devices is not None else base_devices()
        self._resource_interval_s = float(resource_interval_s)
        self._timeline_dir = Path(timeline_dir) if timeline_dir else None

    def run(self) -> List[BenchmarkOutcome]:
        hw = collect_hardware_metadata()
        outcomes: List[BenchmarkOutcome] = []
        for device in self._devices:
            variant = BenchmarkVariant(
                name=f"generation|{device.key}",
                device=device,
                parameters={
                    "n_queries": len(self._queries),
                    "warmup_iterations": self._warmup,
                    "measured_iterations": self._measured,
                    "model_name": getattr(self._generator, "model_name", "unknown"),
                },
            )
            if not getattr(self._generator, "available", False):
                outcomes.append(empty_outcome(
                    "generation", variant,
                    reason=getattr(self._generator, "reason", "generator unavailable"),
                    hardware=hw,
                ))
                continue
            if not device.available:
                outcomes.append(empty_outcome(
                    "generation", variant,
                    reason=device.reason or "device unavailable",
                    hardware=hw,
                ))
                continue
            if device.torch_device.startswith("cuda") and not cuda_is_available():
                outcomes.append(empty_outcome(
                    "generation", variant,
                    reason="CUDA not available at runtime",
                    hardware=hw,
                ))
                continue
            outcomes.append(self._run_variant(variant, device, hw))
        return outcomes

    def _run_variant(
        self,
        variant: BenchmarkVariant,
        device: DeviceSpec,
        hw: HardwareMetadata,
    ) -> BenchmarkOutcome:
        started = _utc_iso()
        wall_start = time.perf_counter()
        warnings: List[str] = []

        reset_gpu_peak_memory()
        timeline = ResourceTimeline(
            interval_s=self._resource_interval_s,
            gpu_monitoring=device.torch_device.startswith("cuda"),
        )

        n = len(self._queries)
        latencies_ms: List[float] = []
        per_call_chars: List[float] = []
        try:
            with timeline:
                # Warmup.
                if n > 0:
                    for i in range(self._warmup):
                        idx = i % n
                        timeline.annotate(f"warmup-{i}-begin")
                        try:
                            self._generator.generate(
                                self._queries[idx], self._contexts[idx],
                            )
                        except Exception as exc:
                            warnings.append(f"warmup failed: {exc}")

                # Measured.
                for i in range(self._measured):
                    if n == 0:
                        break
                    idx = i % n
                    timeline.annotate(f"measured-{i}-begin")
                    t0 = time.perf_counter()
                    try:
                        result = self._generator.generate(
                            self._queries[idx], self._contexts[idx],
                        )
                    except Exception as exc:
                        warnings.append(f"measured {i} failed: {exc}")
                        continue
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies_ms.append(elapsed_ms)
                    answer = ""
                    try:
                        answer = getattr(result, "answer", "") or ""
                    except Exception:
                        pass
                    per_call_chars.append(float(len(answer)))
                    timeline.annotate(f"measured-{i}-end")
        finally:
            pass

        finished = _utc_iso()
        wall_time = time.perf_counter() - wall_start

        primary = summarise(latencies_ms)
        chars_per_sec = items_per_second(
            sum(int(c) for c in per_call_chars),
            sum(latencies_ms) / 1000.0,
        )

        series = [
            make_series("call_latency_ms", latencies_ms, unit="ms"),
            make_series("answer_chars", per_call_chars, unit="chars"),
        ]

        timeline_path = self._write_timeline(timeline, variant)

        return BenchmarkOutcome(
            benchmark="generation",
            variant=variant,
            success=len(latencies_ms) > 0,
            skipped_reason=None if latencies_ms else "no measurements collected",
            started_iso=started,
            finished_iso=finished,
            wall_time_s=round(wall_time, 4),
            warmup_n=self._warmup,
            measured_n=len(latencies_ms),
            measured_unit="ms",
            series=series,
            primary_summary=primary,
            throughput_qps=throughput_qps(latencies_ms),
            items_per_second=chars_per_sec,
            gpu_peak_memory_mb=gpu_peak_memory_mb(),
            resource_timeline_samples=len(timeline.samples),
            resource_timeline_summary=timeline.summary(),
            resource_timeline_path=str(timeline_path) if timeline_path else None,
            hardware=hw,
            metadata={
                "device": device.to_dict(),
                "n_queries": n,
            },
            warnings=warnings,
        )

    def _write_timeline(
        self, timeline: ResourceTimeline, variant: BenchmarkVariant,
    ) -> Optional[Path]:
        if self._timeline_dir is None:
            return None
        safe = variant.name.replace("|", "__").replace(":", "-")
        out_path = self._timeline_dir / f"timeline_{safe}.jsonl"
        try:
            return timeline.write_jsonl(out_path)
        except Exception as exc:
            logger.warning("Failed to write timeline JSONL %s: %s", out_path, exc)
            return None
