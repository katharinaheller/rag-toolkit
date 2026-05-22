"""Embedding benchmarks: CPU vs GPU, batch-size sweep, warm vs cold start.

Runs the real ``rag.embedding`` pipeline through ``rag.embedding.factory``
so the numbers reflect what the production code path actually does. All GPU
paths are guarded — on a CPU-only machine, GPU variants resolve to
``BenchmarkOutcome(success=False, skipped_reason=...)`` without raising.

Determinism: torch / numpy / random seeds are set per variant.
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
    empty_gpu_cache,
    gpu_allocated_memory_mb,
    gpu_peak_memory_mb,
    gpu_synchronize,
    reset_gpu_peak_memory,
    torch_seed_all,
)
from experiments.core.resource_timeline import ResourceTimeline

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class EmbeddingBenchmark:
    """Benchmark a single embedder spec across devices and batch sizes.

    The benchmark intentionally does not reuse ``embedding_adapter.get_embedder``
    because that helper caches a single CPU instance keyed on (spec, mode).
    For an honest CPU-vs-GPU comparison each variant must build its own
    embedder via ``rag.embedding.factory.create_embedder``.
    """

    def __init__(
        self,
        embedder_spec,                       # experiments.configs.default_matrix.EmbedderSpec
        documents: Sequence[str],
        batch_sizes: Sequence[int] = (1, 4, 8, 16, 32),
        warmup_iterations: int = 1,
        measured_iterations: int = 3,
        devices: Optional[Sequence[DeviceSpec]] = None,
        seed: int = 1337,
        resource_interval_s: float = 0.25,
        timeline_dir: Optional[Path] = None,
    ) -> None:
        self._spec = embedder_spec
        self._documents = list(documents)
        self._batch_sizes = [int(b) for b in batch_sizes if int(b) > 0]
        self._warmup = max(0, int(warmup_iterations))
        self._measured = max(1, int(measured_iterations))
        self._devices = list(devices) if devices is not None else base_devices()
        self._seed = int(seed)
        self._resource_interval_s = float(resource_interval_s)
        self._timeline_dir = Path(timeline_dir) if timeline_dir else None

    # ── public entry point ─────────────────────────────────────────────────
    def run(self) -> List[BenchmarkOutcome]:
        hw = collect_hardware_metadata()
        outcomes: List[BenchmarkOutcome] = []
        for device in self._devices:
            for batch_size in self._batch_sizes:
                variant = BenchmarkVariant(
                    name=f"{self._spec.key}|{device.key}|bs={batch_size}",
                    device=device,
                    parameters={
                        "embedder": self._spec.key,
                        "provider": self._spec.provider,
                        "model_name": self._spec.model_name,
                        "batch_size": batch_size,
                        "n_documents": len(self._documents),
                        "warmup_iterations": self._warmup,
                        "measured_iterations": self._measured,
                    },
                )
                if not device.available:
                    outcomes.append(empty_outcome(
                        "embedding", variant,
                        reason=device.reason or "device unavailable",
                        hardware=hw,
                    ))
                    continue
                if device.torch_device.startswith("cuda") and not cuda_is_available():
                    outcomes.append(empty_outcome(
                        "embedding", variant,
                        reason="CUDA not available at runtime",
                        hardware=hw,
                    ))
                    continue
                outcome = self._run_variant(variant, batch_size, device, hw)
                outcomes.append(outcome)
        return outcomes

    # ── one variant ────────────────────────────────────────────────────────
    def _run_variant(
        self,
        variant: BenchmarkVariant,
        batch_size: int,
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

        # Deterministic seeding for reproducibility.
        torch_seed_all(self._seed)

        # Build a fresh embedder for this variant — that's the cold-start cost.
        cfg = EmbeddingConfig(
            provider=self._spec.provider,
            model_name=self._spec.model_name,
            device=device.torch_device,
            batch_size=batch_size,
            max_seq_length=self._spec.max_seq_length,
            use_fp16=self._spec.use_fp16,
            use_bfloat16=self._spec.use_bfloat16,
            retrieval_mode="dense",
            behavior=EmbeddingBehaviorConfig(normalize=True, mode="document"),
            projection=EmbeddingProjectionConfig(target_dim=None),
        )

        # Reset GPU peak counter so the value we report is per-variant.
        reset_gpu_peak_memory()

        cold_t0 = time.perf_counter()
        try:
            embedder = create_embedder(cfg)
        except Exception as exc:
            logger.warning("Embedder build failed (%s, %s): %s",
                           self._spec.key, device.key, exc)
            return empty_outcome(
                "embedding", variant,
                reason=f"embedder build failed: {exc}",
                hardware=hw,
            )
        cold_start_ms = (time.perf_counter() - cold_t0) * 1000.0

        # Resource timeline runs across warmup + measured.
        timeline = ResourceTimeline(
            interval_s=self._resource_interval_s,
            gpu_monitoring=device.torch_device.startswith("cuda"),
        )
        latencies: List[float] = []
        per_doc_ms: List[float] = []
        warm_start_ms: Optional[float] = None
        try:
            with timeline:
                # Warmup.
                for i in range(self._warmup):
                    timeline.annotate(f"warmup-{i}-begin")
                    try:
                        embedder.embed_documents(self._documents[: max(batch_size, 1)])
                    except Exception as exc:
                        warnings.append(f"warmup failed: {exc}")
                        logger.warning("Embedding warmup failed: %s", exc)

                # Mark warm-start (single-batch call right after warmup).
                gpu_synchronize()
                t0 = time.perf_counter()
                try:
                    embedder.embed_documents(self._documents[: max(batch_size, 1)])
                    gpu_synchronize()
                    warm_start_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as exc:
                    warnings.append(f"warm-start measurement failed: {exc}")

                # Measured iterations: embed full document list.
                for i in range(self._measured):
                    timeline.annotate(f"measured-{i}-begin")
                    gpu_synchronize()
                    t0 = time.perf_counter()
                    try:
                        embedder.embed_documents(self._documents)
                        gpu_synchronize()
                    except Exception as exc:
                        warnings.append(f"measurement {i} failed: {exc}")
                        logger.warning("Embedding measurement failed: %s", exc)
                        continue
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    latencies.append(elapsed_ms)
                    if self._documents:
                        per_doc_ms.append(elapsed_ms / max(1, len(self._documents)))
                    timeline.annotate(f"measured-{i}-end")
        finally:
            # Release model + GPU memory.
            try:
                del embedder
            except Exception:
                pass
            gc.collect()
            empty_gpu_cache()

        finished = _utc_iso()
        wall_time = time.perf_counter() - wall_start

        # Stats / outcome.
        primary = summarise(latencies)
        per_doc = summarise(per_doc_ms)
        total_docs = self._measured * len(self._documents)
        ips = items_per_second(total_docs, sum(latencies) / 1000.0)

        series = [
            make_series("call_latency_ms", latencies, unit="ms"),
            make_series("per_document_ms", per_doc_ms, unit="ms",
                        metadata={"n_documents_per_call": len(self._documents)}),
        ]

        timeline_path = self._write_timeline(timeline, variant)
        peak_vram = gpu_peak_memory_mb()
        alloc_vram = gpu_allocated_memory_mb()

        return BenchmarkOutcome(
            benchmark="embedding",
            variant=variant,
            success=len(latencies) > 0,
            skipped_reason=None if latencies else "no measurements collected",
            started_iso=started,
            finished_iso=finished,
            wall_time_s=round(wall_time, 4),
            warmup_n=self._warmup,
            measured_n=len(latencies),
            measured_unit="ms",
            series=series,
            primary_summary=primary,
            throughput_qps=throughput_qps(latencies),
            items_per_second=ips,
            cold_start_ms=round(cold_start_ms, 3),
            warm_start_ms=round(warm_start_ms, 3) if warm_start_ms is not None else None,
            gpu_peak_memory_mb=peak_vram,
            gpu_allocated_memory_mb=alloc_vram,
            resource_timeline_samples=len(timeline.samples),
            resource_timeline_summary=timeline.summary(),
            resource_timeline_path=str(timeline_path) if timeline_path else None,
            hardware=hw,
            metadata={
                "per_document_summary": per_doc.to_dict(),
                "embedder_spec": {
                    "key": self._spec.key,
                    "provider": self._spec.provider,
                    "model_name": self._spec.model_name,
                    "dimension": self._spec.dimension,
                    "use_fp16": self._spec.use_fp16,
                    "use_bfloat16": self._spec.use_bfloat16,
                    "max_seq_length": self._spec.max_seq_length,
                },
                "device": device.to_dict(),
                "n_documents": len(self._documents),
                "seed": self._seed,
            },
            warnings=warnings,
        )

    def _write_timeline(
        self, timeline: ResourceTimeline, variant: BenchmarkVariant,
    ) -> Optional[Path]:
        if self._timeline_dir is None:
            return None
        safe_name = (
            variant.name.replace("|", "__").replace("/", "_").replace(":", "-")
        )
        out_path = self._timeline_dir / f"timeline_{safe_name}.jsonl"
        try:
            return timeline.write_jsonl(out_path)
        except Exception as exc:
            logger.warning("Failed to write timeline JSONL %s: %s", out_path, exc)
            return None
