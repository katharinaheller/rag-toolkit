"""Suite 14 — Throughput and resource benchmarks.

Wraps the rag.evaluation.benchmarks layer to measure retrieval latency
percentiles, throughput, and resident memory for each retriever × corpus pair.
``psutil`` is optional; when missing memory columns degrade to ``None``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from experiments.configs.default_matrix import BENCH_MEASURED, BENCH_WARMUP, CORPORA, RETRIEVERS
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import get_or_build_index
from experiments.adapters.retrieval_adapter import build_retriever, retrieve_timed
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_box_by_group


try:
    import psutil
    _PSUTIL = True
except Exception:
    _PSUTIL = False


_K = 10


def _rss_mb() -> float:
    if not _PSUTIL:
        return 0.0
    try:
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


class ThroughputResourcesSuite(Suite):
    key = "s14_throughput_resources"
    description = "Latency, throughput and memory for each retriever × corpus"

    def run(self) -> Dict[str, Any]:
        agg_rows: List[Dict[str, Any]] = []
        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("benchmark_records.jsonl"))

        try:
            for corpus_name in CORPORA:
                chunks = self.ctx.corpora_chunks.get(corpus_name)
                queries = self.ctx.queries.get(corpus_name, [])
                if not chunks or not queries:
                    continue
                query_texts = [q.text for q in queries[:max(8, BENCH_MEASURED // 2)]]

                for retriever in RETRIEVERS:
                    try:
                        index = get_or_build_index(retriever, chunks, corpus_name, index_cache)
                    except Exception as exc:
                        self.logger.warning("Index build failed (%s/%s): %s",
                                            retriever.key, corpus_name, exc)
                        continue
                    built = build_retriever(retriever, index, top_k=_K)

                    # Warmup.
                    for i in range(BENCH_WARMUP):
                        retrieve_timed(built, query_texts[i % len(query_texts)], k=_K)

                    rss_before = _rss_mb()
                    latencies: List[float] = []
                    for i in range(BENCH_MEASURED):
                        _, ms = retrieve_timed(built, query_texts[i % len(query_texts)], k=_K)
                        latencies.append(ms)
                        raw_writer.write({
                            "retriever": retriever.key,
                            "corpus": corpus_name,
                            "iteration": i,
                            "latency_ms": ms,
                        })
                    rss_after = _rss_mb()

                    latencies.sort()
                    n = len(latencies)
                    mean = sum(latencies) / n if n else 0.0
                    median = latencies[n // 2] if n else 0.0
                    p95 = latencies[max(0, int(0.95 * n) - 1)] if n else 0.0
                    throughput = (n / sum(latencies) * 1000.0) if sum(latencies) > 0 else 0.0

                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "corpus_size": len(chunks),
                        "n_iterations": n,
                        "mean_ms": round(mean, 3),
                        "median_ms": round(median, 3),
                        "p95_ms": round(p95, 3),
                        "throughput_qps": round(throughput, 3),
                        "rss_before_mb": round(rss_before, 1) if _PSUTIL else None,
                        "rss_after_mb": round(rss_after, 1) if _PSUTIL else None,
                        "rss_delta_mb": (round(rss_after - rss_before, 1)
                                         if _PSUTIL else None),
                        "psutil_available": _PSUTIL,
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("throughput_resources.csv"), agg_rows)

        figures: List[str] = []
        # Bucket p95 latency values by retriever for the box-plot.
        grouped_p95: Dict[str, List[float]] = {}
        for row in agg_rows:
            grouped_p95.setdefault(row["retriever"], []).append(float(row["p95_ms"]))
        path = self.figure_path("p95_latency_by_retriever.png")
        if plot_box_by_group(
            grouped_values=grouped_p95,
            out_path=path,
            title="p95 latency by retriever (across corpora)",
            ylabel="p95 latency (ms)",
        ):
            figures.append(str(path))

        findings = _interpret(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("throughput_resources.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No throughput rows produced."]
    largest_corpus = max(rows, key=lambda r: r.get("corpus_size", 0))["corpus"]
    subset = [r for r in rows if r["corpus"] == largest_corpus]
    if subset:
        fastest = min(subset, key=lambda r: r["p95_ms"])
        slowest = max(subset, key=lambda r: r["p95_ms"])
        findings.append(
            f"On {largest_corpus}, fastest retriever (p95) is "
            f"**{fastest['retriever']}** at {fastest['p95_ms']:.1f}ms; slowest is "
            f"**{slowest['retriever']}** at {slowest['p95_ms']:.1f}ms "
            f"(× {slowest['p95_ms'] / max(fastest['p95_ms'], 1e-6):.1f} ratio)."
        )
    # Scaling: how much does p95 grow from smallest to largest corpus?
    by_retr: Dict[str, List[Dict]] = {}
    for r in rows:
        by_retr.setdefault(r["retriever"], []).append(r)
    for retriever, group in by_retr.items():
        group.sort(key=lambda r: r["corpus_size"])
        if len(group) >= 2:
            ratio = group[-1]["p95_ms"] / max(group[0]["p95_ms"], 1e-6)
            findings.append(
                f"**{retriever}** p95 grows × {ratio:.1f} from "
                f"{group[0]['corpus']} ({group[0]['p95_ms']:.1f}ms) to "
                f"{group[-1]['corpus']} ({group[-1]['p95_ms']:.1f}ms)."
            )
    if not _PSUTIL:
        findings.append(
            "psutil not installed — RSS memory columns are null. "
            "`pip install psutil` to enable memory measurement."
        )
    return findings
