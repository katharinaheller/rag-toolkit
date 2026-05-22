"""Suite 19 — Cold start vs warm steady-state.

Quantifies the model-load + first-call penalty against warm steady-state
latency for each embedder on each device. Produces a log-scale grouped-bar
chart so the (often large) cold/warm gap is legible.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.benchmarks import ColdWarmBenchmark
from experiments.benchmarks.base import base_devices
from experiments.configs.default_matrix import (
    COLD_WARM_ITERATIONS,
    COLD_WARM_REPEATS,
    EMBEDDERS,
)
from experiments.configs.settings import SETTINGS
from experiments.suites._bench_shared import (
    hardware_findings,
    largest_corpus,
    persist_outcomes,
    sample_documents,
)
from experiments.suites.base import Suite
from experiments.visualisation.benchmark_plots import plot_cold_vs_warm


class ColdWarmSuite(Suite):
    key = "s19_cold_warm"
    description = "Cold start (load + first call) vs warm steady-state latency"

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
                "findings": ["No corpus available for cold/warm benchmark."],
            }
        docs = sample_documents(chunks, 16)

        all_outcomes = []
        rows: List[Dict[str, Any]] = []

        for spec in EMBEDDERS:
            bench = ColdWarmBenchmark(
                embedder_spec=spec,
                sample_documents=docs,
                warm_iterations=COLD_WARM_ITERATIONS,
                repeats=COLD_WARM_REPEATS,
                devices=base_devices(SETTINGS.gpu_index),
                seed=SETTINGS.seed,
            )
            outcomes = bench.run()
            all_outcomes.extend(outcomes)
            for o in outcomes:
                if not o.success:
                    continue
                warm_mean = o.primary_summary.mean if o.primary_summary else 0.0
                rows.append({
                    "label": f"{spec.key}/{o.variant.device.key}",
                    "embedder": spec.key,
                    "device": o.variant.device.key,
                    "cold_start_ms": o.cold_start_ms,
                    "first_call_ms": o.warm_start_ms,
                    "warm_mean_ms": warm_mean,
                    "cold_over_warm_ratio": o.metadata.get("cold_over_warm_ratio"),
                })

        persist_outcomes(
            all_outcomes,
            self.raw_path("cold_warm_outcomes.jsonl"),
            self.agg_path("cold_warm.csv"),
        )

        figures: List[str] = []
        path = self.figure_path("cold_vs_warm.png")
        if plot_cold_vs_warm(
            rows, path, title="Cold start vs warm steady-state (log scale)",
        ):
            figures.append(str(path))

        findings = self._interpret(rows)
        findings.extend(hardware_findings(all_outcomes))
        return {
            "figures": figures,
            "tables": [str(self.agg_path("cold_warm.csv"))],
            "findings": findings,
            "rows": rows,
        }

    def _interpret(self, rows: List[Dict[str, Any]]) -> List[str]:
        if not rows:
            return ["No cold/warm measurements collected."]
        findings: List[str] = []
        for r in rows:
            ratio = r.get("cold_over_warm_ratio")
            ratio_str = f" (cold is × {ratio:.0f} the warm latency)" if ratio else ""
            findings.append(
                f"**{r['label']}**: cold start {r['cold_start_ms']:.0f}ms vs "
                f"warm {r['warm_mean_ms']:.1f}ms{ratio_str}."
            )
        return findings
