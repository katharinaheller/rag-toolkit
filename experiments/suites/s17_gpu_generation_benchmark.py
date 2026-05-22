"""Suite 17 — CPU vs GPU generation benchmark.

Benchmarks the generator (Ollama) end to end, recording per-call latency,
character throughput and the local resource timeline. When generation is
disabled or unreachable, the suite records a clean skip rather than failing.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.benchmarks import GenerationBenchmark
from experiments.benchmarks.base import base_devices
from experiments.configs.settings import SETTINGS
from experiments.suites._bench_shared import (
    hardware_findings,
    largest_corpus,
    persist_outcomes,
)
from experiments.suites.base import Suite
from experiments.visualisation.benchmark_plots import plot_latency_distribution


class GpuGenerationBenchmarkSuite(Suite):
    key = "s17_gpu_generation_benchmark"
    description = "Generation latency, throughput and resource timeline (CPU/GPU view)"

    _MAX_QUERIES = 8
    _CONTEXTS_PER_QUERY = 4

    def run(self) -> Dict[str, Any]:
        if not SETTINGS.enable_gpu_benchmarks:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["GPU benchmarks disabled via EXPERIMENTS_ENABLE_GPU_BENCHMARKS=0."],
            }
        if not self.ctx.generator.available:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": [
                    "Generator unavailable — generation benchmark skipped: "
                    f"{self.ctx.generator.reason}.",
                ],
            }

        corpus_name, chunks = largest_corpus(self.ctx.corpora_chunks)
        queries = self.ctx.queries.get(corpus_name or "", []) if corpus_name else []
        if not queries:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["No queries available for generation benchmark."],
            }

        # Build contexts per query from the corpus chunk excerpts.
        chunk_texts = [(c.get("text") or "")[:600] for c in chunks if c.get("text")]
        query_texts = [q.text for q in queries[: self._MAX_QUERIES]]
        contexts = [
            chunk_texts[i: i + self._CONTEXTS_PER_QUERY] or chunk_texts[: self._CONTEXTS_PER_QUERY]
            for i in range(len(query_texts))
        ]

        bench = GenerationBenchmark(
            generator=self.ctx.generator,
            queries=query_texts,
            contexts_per_query=contexts,
            warmup_iterations=SETTINGS.benchmark_warmup,
            measured_iterations=max(SETTINGS.benchmark_measured, len(query_texts)),
            devices=base_devices(SETTINGS.gpu_index),
            resource_interval_s=SETTINGS.resource_interval_s,
            timeline_dir=self.raw_dir / "timelines",
        )
        outcomes = bench.run()

        persist_outcomes(
            outcomes,
            self.raw_path("generation_benchmark_outcomes.jsonl"),
            self.agg_path("generation_benchmark.csv"),
        )

        rows: List[Dict[str, Any]] = []
        dist: Dict[str, List[float]] = {}
        for o in outcomes:
            if not o.success or o.primary_summary is None:
                continue
            rows.append({
                "device": o.variant.device.key,
                "mean_latency_ms": o.primary_summary.mean,
                "p95_ms": o.primary_summary.p95,
                "p99_ms": o.primary_summary.p99,
                "throughput_qps": o.throughput_qps,
                "chars_per_second": o.items_per_second,
            })
            for s in o.series:
                if s.label == "call_latency_ms":
                    dist[o.variant.device.key] = s.values

        figures: List[str] = []
        if dist:
            dist_path = self.figure_path("generation_latency_distribution.png")
            if plot_latency_distribution(
                dist, dist_path, title="Generation latency distribution",
            ):
                figures.append(str(dist_path))

        findings = self._interpret(rows)
        findings.extend(hardware_findings(outcomes))
        return {
            "figures": figures,
            "tables": [str(self.agg_path("generation_benchmark.csv"))],
            "findings": findings,
            "rows": rows,
        }

    def _interpret(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return ["No successful generation measurements collected."]
        findings: List[str] = []
        for r in rows:
            findings.append(
                f"Generation on **{r['device']}**: mean "
                f"{r['mean_latency_ms']:.0f}ms, p95 {r['p95_ms']:.0f}ms, "
                f"p99 {r['p99_ms']:.0f}ms, ~{r['chars_per_second']:.0f} chars/s."
            )
        return findings
