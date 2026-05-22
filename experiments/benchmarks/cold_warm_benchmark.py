"""Cold-start vs warm-start benchmark.

Cold start: build the embedder from scratch (model load + first call).
Warm start: model already loaded; subsequent calls only pay inference cost.

Reports both wall-clock times and the speedup ratio (cold/warm). On a CPU-only
host the GPU variant resolves to a skipped outcome cleanly.
"""

from __future__ import annotations

import gc
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
from experiments.core.benchmark_stats import make_series, speedup, summarise
from experiments.core.gpu_hardware import (
    HardwareMetadata,
    collect_hardware_metadata,
    cuda_is_available,
    empty_gpu_cache,
    gpu_peak_memory_mb,
    gpu_synchronize,
    reset_gpu_peak_memory,
    torch_seed_all,
)

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ColdWarmBenchmark:
    """Measure cold-start (model load + first call) vs warm calls per device."""

    def __init__(
        self,
        embedder_spec,
        sample_documents: Sequence[str],
        warm_iterations: int = 10,
        repeats: int = 3,
        devices: Optional[Sequence[DeviceSpec]] = None,
        seed: int = 1337,
    ) -> None:
        self._spec = embedder_spec
        self._docs = list(sample_documents)
        self._warm_iters = max(1, int(warm_iterations))
        self._repeats = max(1, int(repeats))
        self._devices = list(devices) if devices is not None else base_devices()
        self._seed = int(seed)

    def run(self) -> List[BenchmarkOutcome]:
        hw = collect_hardware_metadata()
        outcomes: List[BenchmarkOutcome] = []
        for device in self._devices:
            variant = BenchmarkVariant(
                name=f"cold_warm|{self._spec.key}|{device.key}",
                device=device,
                parameters={
                    "embedder": self._spec.key,
                    "warm_iterations": self._warm_iters,
                    "repeats": self._repeats,
                },
            )
            if not device.available:
                outcomes.append(empty_outcome(
                    "cold_warm", variant,
                    reason=device.reason or "device unavailable",
                    hardware=hw,
                ))
                continue
            if device.torch_device.startswith("cuda") and not cuda_is_available():
                outcomes.append(empty_outcome(
                    "cold_warm", variant,
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
        from rag.embedding.config import (
            EmbeddingBehaviorConfig,
            EmbeddingConfig,
            EmbeddingProjectionConfig,
        )
        from rag.embedding.factory import create_embedder

        started = _utc_iso()
        wall_start = time.perf_counter()
        warnings: List[str] = []

        cold_total_ms: List[float] = []
        first_call_ms: List[float] = []
        warm_call_ms: List[float] = []

        for repeat in range(self._repeats):
            torch_seed_all(self._seed + repeat)
            reset_gpu_peak_memory()

            cfg = EmbeddingConfig(
                provider=self._spec.provider,
                model_name=self._spec.model_name,
                device=device.torch_device,
                batch_size=self._spec.batch_size,
                max_seq_length=self._spec.max_seq_length,
                use_fp16=self._spec.use_fp16,
                use_bfloat16=self._spec.use_bfloat16,
                retrieval_mode="dense",
                behavior=EmbeddingBehaviorConfig(normalize=True, mode="document"),
                projection=EmbeddingProjectionConfig(target_dim=None),
            )

            # Cold = build + first call.
            t_build = time.perf_counter()
            try:
                embedder = create_embedder(cfg)
            except Exception as exc:
                warnings.append(f"repeat {repeat}: build failed: {exc}")
                continue
            gpu_synchronize()

            t_first = time.perf_counter()
            try:
                embedder.embed_documents(self._docs[: min(len(self._docs), 4)])
                gpu_synchronize()
            except Exception as exc:
                warnings.append(f"repeat {repeat}: first call failed: {exc}")
                try:
                    del embedder
                except Exception:
                    pass
                continue
            first_done = time.perf_counter()

            cold_total_ms.append((first_done - t_build) * 1000.0)
            first_call_ms.append((first_done - t_first) * 1000.0)

            # Warm calls.
            for _ in range(self._warm_iters):
                gpu_synchronize()
                t0 = time.perf_counter()
                try:
                    embedder.embed_documents(self._docs[: min(len(self._docs), 4)])
                    gpu_synchronize()
                except Exception as exc:
                    warnings.append(f"warm call failed: {exc}")
                    continue
                warm_call_ms.append((time.perf_counter() - t0) * 1000.0)

            try:
                del embedder
            except Exception:
                pass
            gc.collect()
            empty_gpu_cache()

        finished = _utc_iso()
        wall_time = time.perf_counter() - wall_start

        warm_summary = summarise(warm_call_ms)
        cold_avg = (sum(cold_total_ms) / len(cold_total_ms)) if cold_total_ms else 0.0
        first_avg = (sum(first_call_ms) / len(first_call_ms)) if first_call_ms else 0.0

        return BenchmarkOutcome(
            benchmark="cold_warm",
            variant=variant,
            success=bool(cold_total_ms) and bool(warm_call_ms),
            skipped_reason=None,
            started_iso=started,
            finished_iso=finished,
            wall_time_s=round(wall_time, 4),
            warmup_n=0,
            measured_n=len(warm_call_ms),
            measured_unit="ms",
            series=[
                make_series("cold_total_ms", cold_total_ms, unit="ms",
                            metadata={"definition": "build + first call"}),
                make_series("first_call_ms", first_call_ms, unit="ms",
                            metadata={"definition": "first inference after build"}),
                make_series("warm_call_ms", warm_call_ms, unit="ms",
                            metadata={"definition": "subsequent steady-state calls"}),
            ],
            primary_summary=warm_summary,
            cold_start_ms=round(cold_avg, 3),
            warm_start_ms=round(first_avg, 3),
            gpu_peak_memory_mb=gpu_peak_memory_mb(),
            hardware=hw,
            metadata={
                "embedder_spec": {
                    "key": self._spec.key,
                    "provider": self._spec.provider,
                    "model_name": self._spec.model_name,
                },
                "device": device.to_dict(),
                "cold_over_warm_ratio": (
                    speedup(warm_summary.mean, cold_avg) if warm_summary.mean else None
                ),
                "first_call_over_warm_ratio": (
                    speedup(warm_summary.mean, first_avg) if warm_summary.mean else None
                ),
                "n_repeats": self._repeats,
                "warm_iterations_per_repeat": self._warm_iters,
                "seed": self._seed,
            },
            warnings=warnings,
        )
