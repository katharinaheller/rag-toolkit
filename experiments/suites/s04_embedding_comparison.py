"""Suite 4 — EmbeddingGemma vs BGE-M3 comparison.

Direct head-to-head: only the embedding model varies, retrieval mode is held
constant (dense, then hybrid). Reports recall, latency and discriminability
(score spread) per corpus.
"""

from __future__ import annotations

import statistics as stats
from typing import Any, Dict, List

from experiments.configs.default_matrix import CORPORA, RETRIEVERS_BY_KEY, TOPK_VALUES
from experiments.metrics import aggregate_retrieval_metrics
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_retriever_corpus_scaling


_K = 10
_PAIRS = [
    ("dense_gemma", "dense_bge"),
    ("hybrid_gemma", "hybrid_bge"),
]


class EmbeddingComparisonSuite(Suite):
    key = "s04_embedding_comparison"
    description = "Head-to-head: EmbeddingGemma vs BGE-M3"

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
                gold = gold_lookup_chunks(queries)

                for retriever_key in ("dense_gemma", "dense_bge",
                                      "hybrid_gemma", "hybrid_bge"):
                    if retriever_key not in RETRIEVERS_BY_KEY:
                        continue
                    retriever = RETRIEVERS_BY_KEY[retriever_key]
                    try:
                        index = get_or_build_index(retriever, chunks, corpus_name, index_cache)
                    except Exception as exc:
                        self.logger.warning(
                            "Index build failed (%s/%s): %s",
                            retriever.key, corpus_name, exc,
                        )
                        continue

                    records = run_retriever_on_queries(
                        run_id=self.ctx.run_id,
                        suite_key=self.key,
                        retriever=retriever,
                        built_index=index,
                        corpus_name=corpus_name,
                        queries=queries,
                        top_k=max(TOPK_VALUES),
                    )
                    raw_writer.write_many(records)

                    metrics = aggregate_retrieval_metrics(
                        [r.to_dict() for r in records],
                        gold,
                        k_values=TOPK_VALUES,
                    )
                    score_spread = _score_spread([r.score for r in records[:200]])
                    mean_latency = (
                        sum(r.latency_ms for r in records) / len(records)
                        if records else 0.0
                    )

                    row: Dict[str, Any] = {
                        "retriever": retriever.key,
                        "embedder": retriever.embedder_key,
                        "corpus": corpus_name,
                        "corpus_size": len(chunks),
                        "score_spread": round(score_spread, 6),
                        "mean_latency_ms": round(mean_latency, 3),
                    }
                    row.update({m: round(v, 6) for m, v in metrics.items()})
                    agg_rows.append(row)
        finally:
            raw_writer.close()

        write_csv(self.agg_path("embedding_comparison.csv"), agg_rows)

        figures: List[str] = []
        for metric in (f"recall@{_K}", f"ndcg@{_K}"):
            fig = self.figure_path(f"{metric}_embedder_vs_corpus.png")
            if plot_retriever_corpus_scaling(
                data=agg_rows, metric=metric, out_path=fig,
                title=f"{metric} — Gemma vs BGE",
            ):
                figures.append(str(fig))

        findings = _interpret_embedding(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("embedding_comparison.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _score_spread(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    try:
        return stats.pstdev(values)
    except stats.StatisticsError:
        return 0.0


def _interpret_embedding(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No embedding-comparison rows produced."]
    for pair in _PAIRS:
        a, b = pair
        for corpus in sorted({r["corpus"] for r in rows}):
            ra = next((r for r in rows if r["retriever"] == a and r["corpus"] == corpus), None)
            rb = next((r for r in rows if r["retriever"] == b and r["corpus"] == corpus), None)
            if ra is None or rb is None:
                continue
            metric_key = f"recall@{_K}"
            delta = ra.get(metric_key, 0.0) - rb.get(metric_key, 0.0)
            winner, loser = (a, b) if delta > 0 else (b, a)
            findings.append(
                f"On {corpus}: **{winner}** beats **{loser}** by {abs(delta):.3f} "
                f"on {metric_key}. Score spread (winner): "
                f"{(ra if delta > 0 else rb)['score_spread']:.3f}; "
                f"latency (ms): {ra['mean_latency_ms']:.1f} vs {rb['mean_latency_ms']:.1f}."
            )
    return findings
