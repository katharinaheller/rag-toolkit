"""Suite 3 — Dense vs sparse vs hybrid stratified by query type.

Question: Under which query types does each retriever family dominate?
Specifically: does TF-IDF beat dense on keyword queries? Does dense beat
sparse on paraphrases? Are noisy queries the regime where hybrid wins?
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS
from experiments.core.types import QUERY_TYPES
from experiments.metrics import aggregate_retrieval_metrics
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_query_type_grouped_bars


_K = 10
_FALLBACK_CORPUS_ORDER = ["n1000", "n100", "n50", "n10"]


class QueryTypeSuite(Suite):
    key = "s03_query_type_comparison"
    description = "Dense vs sparse vs hybrid stratified by query type"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for query-type suite.")

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]

        agg_rows: List[Dict[str, Any]] = []
        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("retrieval_records.jsonl"))

        try:
            for retriever in RETRIEVERS:
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
                    top_k=_K,
                )
                raw_writer.write_many(records)

                gold = gold_lookup_chunks(queries)
                # Slice per query_type.
                for qtype in QUERY_TYPES:
                    qids = {q.query_id for q in queries if q.query_type == qtype}
                    if not qids:
                        continue
                    subset = [r.to_dict() for r in records if r.query_id in qids]
                    if not subset:
                        continue
                    subset_gold = {qid: g for qid, g in gold.items() if qid in qids}
                    metrics = aggregate_retrieval_metrics(
                        subset, subset_gold, k_values=[_K],
                    )
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "query_type": qtype,
                        "n_queries": len(qids),
                        "recall": round(metrics[f"recall@{_K}"], 6),
                        "precision": round(metrics[f"precision@{_K}"], 6),
                        "ndcg": round(metrics[f"ndcg@{_K}"], 6),
                        "mrr": round(metrics[f"mrr@{_K}"], 6),
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("by_query_type.csv"), agg_rows)

        figures: List[str] = []
        for metric in ("recall", "ndcg", "mrr"):
            path = self.figure_path(f"{metric}_by_query_type.png")
            if plot_query_type_grouped_bars(
                data=agg_rows, metric=metric, out_path=path,
                title=f"{metric}@{_K} by query type ({corpus_name})",
            ):
                figures.append(str(path))

        findings = _interpret_by_type(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("by_query_type.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_by_type(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["No query-type rows produced."]
    findings: List[str] = []
    qtypes = sorted({r["query_type"] for r in rows})
    for qt in qtypes:
        subset = sorted(
            (r for r in rows if r["query_type"] == qt),
            key=lambda r: r["recall"], reverse=True,
        )
        if not subset:
            continue
        best = subset[0]
        if len(subset) > 1:
            findings.append(
                f"On **{qt}** queries, the best retriever is **{best['retriever']}** "
                f"(recall@{_K}={best['recall']:.3f}); margin over second-best: "
                f"{best['recall'] - subset[1]['recall']:.3f}."
            )
        else:
            findings.append(
                f"On **{qt}** queries, only **{best['retriever']}** produced results "
                f"(recall@{_K}={best['recall']:.3f})."
            )
    return findings
