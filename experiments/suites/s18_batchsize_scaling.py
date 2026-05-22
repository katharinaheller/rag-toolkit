"""Suite 18 — Batch-size throughput scaling.

Sweeps the embedding batch size on every available device and plots
throughput (documents/second) and peak VRAM against batch size. Reveals the
batch size at which throughput saturates and the VRAM cost of larger batches.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.benchmarks import EmbeddingBenchmark
from experiments.benchmarks.base import base_devices
from experiments.configs.default_matrix import BENCH_BATCH_SIZES, EMBEDDERS
from experiments.configs.settings import SETTINGS
from experiments.suites._bench_shared import (
    hardware_findings,
    largest_corpus,
    persist_outcomes,
    sample_documents,
)
from experiments.suites.base import Suite
from experiments.visualisation.benchmark_plots import plot_throughput_vs_batchsize
from experiments.visualisation.resource_plots import plot_vram_vs_batchsize


class BatchsizeScalingSuite(Suite):
    key = "s18_batchsize_scaling"
    description = "Embedding throughput and VRAM as a function of batch size"

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
                "findings": ["No corpus available for batch-size scaling."],
            }
        n_docs = max(SETTINGS.benchmark_doc_sample, max(BENCH_BATCH_SIZES))
        docs = sample_documents(chunks, n_docs)

        # Benchmark only the primary embedder to keep runtime bounded; the
        # batch-scaling shape is embedder-independent enough for the thesis.
        spec = EMBEDDERS[0]
        bench = EmbeddingBenchmark(
            embedder_spec=spec,
            documents=docs,
            batch_sizes=BENCH_BATCH_SIZES,
            warmup_iterations=SETTINGS.benchmark_warmup,
            measured_iterations=SETTINGS.benchmark_measured,
            devices=base_devices(SETTINGS.gpu_index),
            seed=SETTINGS.seed,
            resource_interval_s=SETTINGS.resource_interval_s,
            timeline_dir=self.raw_dir / "timelines",
        )
        outcomes = bench.run()
        persist_outcomes(
            outcomes,
            self.raw_path("batchsize_outcomes.jsonl"),
            self.agg_path("batchsize_scaling.csv"),
        )

        rows: List[Dict[str, Any]] = []
        for o in outcomes:
            if not o.success:
                continue
            rows.append({
                "device": o.variant.device.key,
                "batch_size": o.variant.parameters.get("batch_size"),
                "items_per_second": o.items_per_second,
                "mean_latency_ms": o.primary_summary.mean if o.primary_summary else None,
                "gpu_peak_memory_mb": o.gpu_peak_memory_mb,
            })

        figures: List[str] = []
        tput_path = self.figure_path("throughput_vs_batchsize.png")
        if plot_throughput_vs_batchsize(
            rows, tput_path,
            title=f"Throughput vs batch size ({spec.key})",
        ):
            figures.append(str(tput_path))

        vram_path = self.figure_path("vram_vs_batchsize.png")
        if plot_vram_vs_batchsize(
            rows, vram_path,
            title=f"Peak VRAM vs batch size ({spec.key})",
        ):
            figures.append(str(vram_path))

        findings = self._interpret(rows, spec.key)
        findings.extend(hardware_findings(outcomes))
        return {
            "figures": figures,
            "tables": [str(self.agg_path("batchsize_scaling.csv"))],
            "findings": findings,
            "rows": rows,
        }

    def _interpret(self, rows: List[Dict[str, Any]], embedder: str) -> List[str]:
        if not rows:
            return ["No batch-size scaling measurements collected."]
        findings: List[str] = []
        by_device: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            if r.get("items_per_second") is None:
                continue
            by_device.setdefault(r["device"], []).append(r)
        for device, group in by_device.items():
            group.sort(key=lambda r: r["batch_size"])
            best = max(group, key=lambda r: r["items_per_second"])
            findings.append(
                f"**{device}** ({embedder}): peak throughput "
                f"{best['items_per_second']:.0f} docs/s at batch="
                f"{best['batch_size']}."
            )
            if len(group) >= 2 and group[0]["items_per_second"] > 0:
                ratio = best["items_per_second"] / group[0]["items_per_second"]
                findings.append(
                    f"**{device}**: batching from {group[0]['batch_size']} to "
                    f"{best['batch_size']} improves throughput × {ratio:.1f}."
                )
        return findings
