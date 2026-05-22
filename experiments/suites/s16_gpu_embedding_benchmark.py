"""Suite 16 — CPU vs GPU embedding benchmark.

Runs the embedding pipeline for each embedder spec on CPU and (when available)
GPU at a single representative batch size, measuring per-call latency,
throughput, cold-start cost and peak VRAM. Produces a CPU-vs-GPU comparison
bar chart and a speedup chart.

Graceful on CPU-only hosts: GPU rows are recorded as "skipped".
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.benchmarks import EmbeddingBenchmark
from experiments.benchmarks.base import base_devices
from experiments.configs.default_matrix import EMBEDDERS
from experiments.configs.settings import SETTINGS
from experiments.core.benchmark_stats import speedup
from experiments.suites._bench_shared import (
    hardware_findings,
    largest_corpus,
    persist_outcomes,
    sample_documents,
)
from experiments.suites.base import Suite
from experiments.visualisation.benchmark_plots import plot_cpu_gpu_comparison
from experiments.visualisation.resource_plots import plot_speedup_bars


class GpuEmbeddingBenchmarkSuite(Suite):
    key = "s16_gpu_embedding_benchmark"
    description = "CPU vs GPU embedding latency, throughput, VRAM and cold start"

    _REPRESENTATIVE_BATCH = 16

    def run(self) -> Dict[str, Any]:
        if not SETTINGS.enable_gpu_benchmarks:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["GPU benchmarks disabled via EXPERIMENTS_ENABLE_GPU_BENCHMARKS=0."],
            }

        corpus_name, chunks = largest_corpus(self.ctx.corpora_chunks)
        if not chunks:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["No corpus available for embedding benchmark."],
            }
        docs = sample_documents(chunks, SETTINGS.benchmark_doc_sample)

        all_outcomes = []
        comparison_rows: List[Dict[str, Any]] = []
        speedup_rows: List[Dict[str, Any]] = []

        for spec in EMBEDDERS:
            bench = EmbeddingBenchmark(
                embedder_spec=spec,
                documents=docs,
                batch_sizes=[self._REPRESENTATIVE_BATCH],
                warmup_iterations=SETTINGS.benchmark_warmup,
                measured_iterations=SETTINGS.benchmark_measured,
                devices=base_devices(SETTINGS.gpu_index),
                seed=SETTINGS.seed,
                resource_interval_s=SETTINGS.resource_interval_s,
                timeline_dir=self.raw_dir / "timelines",
            )
            outcomes = bench.run()
            all_outcomes.extend(outcomes)

            cpu_mean = None
            gpu_mean = None
            for o in outcomes:
                if not o.success or o.primary_summary is None:
                    continue
                row = {
                    "workload": spec.key,
                    "device": o.variant.device.key,
                    "mean_latency_ms": o.primary_summary.mean,
                    "items_per_second": o.items_per_second,
                    "gpu_peak_memory_mb": o.gpu_peak_memory_mb,
                }
                comparison_rows.append(row)
                if o.variant.device.key == "cpu":
                    cpu_mean = o.primary_summary.mean
                elif o.variant.device.key.startswith("cuda"):
                    gpu_mean = o.primary_summary.mean
            if cpu_mean is not None and gpu_mean is not None:
                speedup_rows.append({
                    "label": spec.key,
                    "speedup": speedup(cpu_mean, gpu_mean),
                })

        persist_outcomes(
            all_outcomes,
            self.raw_path("embedding_benchmark_outcomes.jsonl"),
            self.agg_path("embedding_benchmark.csv"),
        )

        figures: List[str] = []
        cmp_path = self.figure_path("cpu_gpu_embedding_latency.png")
        if plot_cpu_gpu_comparison(
            comparison_rows, cmp_path,
            metric_field="mean_latency_ms",
            title="CPU vs GPU embedding latency (batch=16)",
            ylabel="Mean call latency (ms)",
            group_field="workload",
        ):
            figures.append(str(cmp_path))

        if speedup_rows:
            sp_path = self.figure_path("embedding_speedup.png")
            if plot_speedup_bars(
                speedup_rows, sp_path,
                title="GPU embedding speedup over CPU",
            ):
                figures.append(str(sp_path))

        findings = self._interpret(comparison_rows, speedup_rows)
        findings.extend(hardware_findings(all_outcomes))

        return {
            "figures": figures,
            "tables": [str(self.agg_path("embedding_benchmark.csv"))],
            "findings": findings,
            "rows": comparison_rows,
        }

    def _interpret(
        self, rows: List[Dict[str, Any]], speedup_rows: List[Dict[str, Any]],
    ) -> List[str]:
        findings: List[str] = []
        if not rows:
            return ["No successful embedding benchmark measurements."]
        for sp in speedup_rows:
            factor = sp["speedup"]
            verb = "faster" if factor >= 1.0 else "slower"
            findings.append(
                f"**{sp['label']}**: GPU is {factor:.2f}× {verb} than CPU "
                f"for batch={self._REPRESENTATIVE_BATCH} embedding."
            )
        gpu_rows = [r for r in rows if r["device"].startswith("cuda")
                    and r.get("gpu_peak_memory_mb")]
        for r in gpu_rows:
            findings.append(
                f"**{r['workload']}** peak VRAM on GPU: "
                f"{r['gpu_peak_memory_mb']:.0f} MiB."
            )
        if not any(r["device"].startswith("cuda") for r in rows):
            findings.append(
                "No GPU measurements were collected (CUDA unavailable); "
                "only CPU embedding latencies are reported."
            )
        return findings
