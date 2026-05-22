"""Suite 21 — Retrieval concurrency scaling.

Measures wall-clock throughput and p95 latency of a representative retriever
as the number of concurrent worker threads increases. Reveals whether the
CPU-bound retrieval path scales with threads or is serialised by the GIL /
shared index locks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.adapters.retrieval_adapter import build_retriever, retrieve_timed
from experiments.benchmarks import ConcurrencyBenchmark
from experiments.benchmarks.base import DeviceSpec
from experiments.configs.default_matrix import (
    BENCH_CONCURRENCY_LEVELS,
    RETRIEVERS_BY_KEY,
)
from experiments.configs.settings import SETTINGS
from experiments.suites._bench_shared import (
    hardware_findings,
    largest_corpus,
    persist_outcomes,
)
from experiments.suites._shared import get_or_build_index
from experiments.suites.base import Suite
from experiments.visualisation.benchmark_plots import plot_concurrency_scaling


class ConcurrencyScalingSuite(Suite):
    key = "s21_concurrency_scaling"
    description = "Retrieval throughput and p95 latency vs concurrency level"

    _RETRIEVER_KEY = "bm25"   # cheap, deterministic, no model load
    _TOP_K = 10
    _TASKS_PER_LEVEL = 64

    def run(self) -> Dict[str, Any]:
        if not SETTINGS.enable_gpu_benchmarks:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["GPU benchmarks disabled via EXPERIMENTS_ENABLE_GPU_BENCHMARKS=0."],
            }

        corpus_name, chunks = largest_corpus(self.ctx.corpora_chunks)
        queries = self.ctx.queries.get(corpus_name or "", []) if corpus_name else []
        if not chunks or not queries:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": ["No corpus/queries available for concurrency scaling."],
            }

        retriever_spec = RETRIEVERS_BY_KEY.get(self._RETRIEVER_KEY)
        if retriever_spec is None:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": [f"Retriever '{self._RETRIEVER_KEY}' not in matrix."],
            }

        index_cache: Dict = {}
        try:
            index = get_or_build_index(
                retriever_spec, chunks, corpus_name, index_cache,
            )
        except Exception as exc:
            return {
                "figures": [], "tables": [], "rows": [],
                "findings": [f"Index build failed for concurrency suite: {exc}"],
            }
        built = build_retriever(retriever_spec, index, top_k=self._TOP_K)

        query_texts = [q.text for q in queries]

        def _workload(query: str) -> Any:
            return retrieve_timed(built, query, k=self._TOP_K)

        bench = ConcurrencyBenchmark(
            name=f"retrieval-{self._RETRIEVER_KEY}",
            workload=_workload,
            payloads=query_texts,
            concurrency_levels=BENCH_CONCURRENCY_LEVELS,
            tasks_per_level=self._TASKS_PER_LEVEL,
            warmup_tasks=8,
            device=DeviceSpec(
                key="cpu", torch_device="cpu", label="CPU", available=True,
            ),
            resource_interval_s=SETTINGS.resource_interval_s,
            timeline_dir=self.raw_dir / "timelines",
        )
        outcomes = bench.run()
        persist_outcomes(
            outcomes,
            self.raw_path("concurrency_outcomes.jsonl"),
            self.agg_path("concurrency_scaling.csv"),
        )

        rows: List[Dict[str, Any]] = []
        for o in outcomes:
            if not o.success or o.primary_summary is None:
                continue
            rows.append({
                "workload": self._RETRIEVER_KEY,
                "concurrency": o.variant.parameters.get("concurrency"),
                "throughput_qps": o.throughput_qps,
                "p95_ms": o.primary_summary.p95,
                "mean_ms": o.primary_summary.mean,
            })

        figures: List[str] = []
        path = self.figure_path("concurrency_scaling.png")
        if plot_concurrency_scaling(
            rows, path, title=f"Concurrency scaling ({self._RETRIEVER_KEY})",
        ):
            figures.append(str(path))

        findings = self._interpret(rows)
        findings.extend(hardware_findings(outcomes))
        return {
            "figures": figures,
            "tables": [str(self.agg_path("concurrency_scaling.csv"))],
            "findings": findings,
            "rows": rows,
        }

    def _interpret(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return ["No concurrency measurements collected."]
        rows = sorted(rows, key=lambda r: r["concurrency"])
        base = rows[0]
        best = max(rows, key=lambda r: r["throughput_qps"])
        findings: List[str] = [
            f"Throughput at concurrency=1: {base['throughput_qps']:.1f} qps; "
            f"peak {best['throughput_qps']:.1f} qps at concurrency="
            f"{best['concurrency']}."
        ]
        if base["throughput_qps"] > 0:
            scaling = best["throughput_qps"] / base["throughput_qps"]
            ideal = best["concurrency"]
            efficiency = (scaling / ideal * 100.0) if ideal else 0.0
            findings.append(
                f"Parallel scaling factor × {scaling:.2f} at "
                f"{best['concurrency']} workers "
                f"({efficiency:.0f}% of ideal linear scaling)."
            )
        return findings
