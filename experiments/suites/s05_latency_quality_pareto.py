"""Suite 5 — Latency-quality Pareto frontier.

Pairs mean retrieval latency with recall@k for each retriever × corpus and
identifies the non-dominated set (minimise latency, maximise recall).
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import CORPORA, RETRIEVERS, TOPK_VALUES
from rag.evaluation.metrics import (
    ParetoPoint,
    aggregate_retrieval_metrics,
    compute_pareto_front,
)
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_pareto, plot_latency_quality_scatter


_K = 10


class LatencyQualityParetoSuite(Suite):
    key = "s05_latency_quality_pareto"
    description = "Latency vs quality trade-offs and Pareto frontier"

    def run(self) -> Dict[str, Any]:
        agg_rows: List[Dict[str, Any]] = []
        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("retrieval_records.jsonl"))

        try:
            for corpus_name in CORPORA:
                chunks = self.ctx.corpora_chunks.get(corpus_name)
                queries = self.ctx.queries.get(corpus_name, [])
                if not chunks or not queries:
                    continue

                for retriever in RETRIEVERS:
                    try:
                        index = get_or_build_index(retriever, chunks, corpus_name, index_cache)
                    except Exception as exc:
                        self.logger.warning("Index build failed (%s/%s): %s",
                                            retriever.key, corpus_name, exc)
                        continue

                    records = run_retriever_on_queries(
                        run_id=self.ctx.run_id, suite_key=self.key,
                        retriever=retriever, built_index=index,
                        corpus_name=corpus_name, queries=queries, top_k=_K,
                    )
                    raw_writer.write_many(records)
                    if not records:
                        continue

                    gold = gold_lookup_chunks(queries)
                    metrics = aggregate_retrieval_metrics(
                        [r.to_dict() for r in records], gold, k_values=[_K],
                    )
                    latencies = [r.latency_ms for r in records]
                    mean_lat = sum(latencies) / len(latencies)
                    sorted_lat = sorted(latencies)
                    p95 = sorted_lat[max(0, int(0.95 * len(sorted_lat)) - 1)]

                    agg_rows.append({
                        "retriever": retriever.key,
                        "embedder": retriever.embedder_key,
                        "corpus": corpus_name,
                        "corpus_size": len(chunks),
                        "mean_latency_ms": round(mean_lat, 3),
                        "p95_latency_ms": round(p95, 3),
                        f"recall@{_K}": round(metrics[f"recall@{_K}"], 6),
                        f"ndcg@{_K}": round(metrics[f"ndcg@{_K}"], 6),
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("latency_quality.csv"), agg_rows)

        # Pareto: minimise latency, maximise recall.
        pareto_rows: List[Dict[str, Any]] = []
        front_by_corpus: Dict[str, List[ParetoPoint]] = {}
        for corpus in sorted({r["corpus"] for r in agg_rows}):
            subset = [r for r in agg_rows if r["corpus"] == corpus]
            pts = [
                ParetoPoint(
                    label=r["retriever"],
                    x=r["mean_latency_ms"],
                    y=r[f"recall@{_K}"],
                    extras={"corpus": corpus},
                )
                for r in subset
            ]
            front = compute_pareto_front(pts)
            front_by_corpus[corpus] = front
            for p in front:
                pareto_rows.append({
                    "corpus": corpus, "retriever": p.label,
                    "latency_ms": p.x, f"recall@{_K}": p.y, "pareto": True,
                })
        write_csv(self.agg_path("pareto_front.csv"), pareto_rows)

        figures: List[str] = []
        scatter_fig = self.figure_path("latency_vs_recall_scatter.png")
        if plot_latency_quality_scatter(
            points=agg_rows,
            out_path=scatter_fig,
            title=f"Latency vs recall@{_K}",
            x_field="mean_latency_ms",
            y_field=f"recall@{_K}",
        ):
            figures.append(str(scatter_fig))

        for corpus, front in front_by_corpus.items():
            subset = [r for r in agg_rows if r["corpus"] == corpus]
            scatter_points = [
                {
                    "label": r["retriever"],
                    "retriever": r["retriever"],
                    "latency_ms": r["mean_latency_ms"],
                    "recall": r[f"recall@{_K}"],
                }
                for r in subset
            ]
            front_labels = [p.label for p in front]
            fig = self.figure_path(f"pareto_{corpus}.png")
            if plot_pareto(
                points=scatter_points,
                front_labels=front_labels,
                out_path=fig,
                x_field="latency_ms",
                y_field="recall",
                title=f"Pareto front ({corpus}) — latency vs recall@{_K}",
                x_label="mean latency (ms)",
                y_label=f"recall@{_K}",
                label_field="label",
            ):
                figures.append(str(fig))

        findings = _interpret(agg_rows, pareto_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("latency_quality.csv")),
                       str(self.agg_path("pareto_front.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret(rows: List[Dict[str, Any]], pareto: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No latency-quality rows produced."]
    for corpus in sorted({r["corpus"] for r in rows}):
        front = [p["retriever"] for p in pareto if p["corpus"] == corpus]
        if front:
            findings.append(
                f"On {corpus}, the Pareto-optimal retrievers (latency vs recall@{_K}) are: "
                f"**{', '.join(front)}**."
            )
        subset = [r for r in rows if r["corpus"] == corpus]
        if subset:
            fastest = min(subset, key=lambda r: r["mean_latency_ms"])
            best = max(subset, key=lambda r: r[f"recall@{_K}"])
            if fastest["retriever"] != best["retriever"]:
                gain = best[f"recall@{_K}"] - fastest[f"recall@{_K}"]
                cost = best["mean_latency_ms"] - fastest["mean_latency_ms"]
                findings.append(
                    f"On {corpus}, moving from fastest (**{fastest['retriever']}**, "
                    f"{fastest['mean_latency_ms']:.1f}ms) to best-quality "
                    f"(**{best['retriever']}**, {best['mean_latency_ms']:.1f}ms) "
                    f"buys {gain:.3f} recall@{_K} at +{cost:.1f}ms cost."
                )
    return findings
