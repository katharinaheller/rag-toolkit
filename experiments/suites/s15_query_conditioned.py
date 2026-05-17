"""Suite 15 — Query-conditioned configuration analysis.

Picks the best retriever per query type and asks: what would the recall be if
we routed each query to its optimum retriever (oracle routing)? The gap
between oracle and the best single retriever is the upper bound on query-type
adaptive selection.
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
_FALLBACK_CORPUS_ORDER = ["n100", "n1000", "n50", "n10"]


class QueryConditionedSuite(Suite):
    key = "s15_query_conditioned"
    description = "Best retriever per query type and oracle routing upper bound"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for query-conditioned suite.")

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]
        gold = gold_lookup_chunks(queries)

        # For each retriever × query_type, mean recall.
        results: Dict[str, Dict[str, float]] = {}
        per_query_recall: Dict[str, Dict[str, float]] = {}  # query_id -> retriever -> recall

        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("retrieval_records.jsonl"))

        try:
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

                # Per-query recall@k.
                per_query: Dict[str, List[str]] = {}
                for r in records:
                    per_query.setdefault(r.query_id, []).append(r.chunk_id)
                for q in queries:
                    retrieved = per_query.get(q.query_id, [])[:_K]
                    gset = gold.get(q.query_id, set())
                    if gset:
                        rec = len(set(retrieved) & gset) / len(gset)
                    else:
                        rec = 0.0
                    per_query_recall.setdefault(q.query_id, {})[retriever.key] = rec

                # Per query-type mean recall.
                results[retriever.key] = {}
                for qtype in QUERY_TYPES:
                    qids = [q.query_id for q in queries if q.query_type == qtype]
                    if not qids:
                        continue
                    recs = [per_query_recall.get(qid, {}).get(retriever.key, 0.0)
                            for qid in qids]
                    results[retriever.key][qtype] = (sum(recs) / len(recs)) if recs else 0.0
        finally:
            raw_writer.close()

        # Best retriever per query type and oracle.
        type_rows: List[Dict[str, Any]] = []
        best_per_type: Dict[str, str] = {}
        for qtype in QUERY_TYPES:
            ranking = sorted(
                ((r, results[r].get(qtype, 0.0)) for r in results),
                key=lambda t: t[1], reverse=True,
            )
            if not ranking:
                continue
            best_per_type[qtype] = ranking[0][0]
            for retriever_key, value in ranking:
                type_rows.append({
                    "query_type": qtype,
                    "retriever": retriever_key,
                    "recall": round(value, 6),
                })

        write_csv(self.agg_path("recall_by_type_per_retriever.csv"), type_rows)

        # Single best retriever overall (avg over query types).
        overall_means = {r: sum(results[r].values()) / max(1, len(results[r]))
                         for r in results}
        best_overall = max(overall_means, key=overall_means.get) if overall_means else None

        # Oracle routing per-query: pick max recall among retrievers per query.
        oracle_per_query: List[float] = []
        single_best: List[float] = []
        for q in queries:
            recs = per_query_recall.get(q.query_id, {})
            if not recs:
                continue
            oracle_per_query.append(max(recs.values()))
            if best_overall is not None:
                single_best.append(recs.get(best_overall, 0.0))

        oracle_recall = sum(oracle_per_query) / len(oracle_per_query) if oracle_per_query else 0.0
        single_best_recall = sum(single_best) / len(single_best) if single_best else 0.0

        summary_rows = [{
            "retriever": "ORACLE_PER_QUERY",
            "scope": "all",
            "recall": round(oracle_recall, 6),
            "n_queries": len(oracle_per_query),
        }]
        if best_overall is not None:
            summary_rows.append({
                "retriever": best_overall,
                "scope": "best_single",
                "recall": round(single_best_recall, 6),
                "n_queries": len(single_best),
            })
        for qtype, retriever_key in best_per_type.items():
            summary_rows.append({
                "retriever": retriever_key,
                "scope": f"best_for_{qtype}",
                "recall": round(results[retriever_key][qtype], 6),
                "n_queries": sum(1 for q in queries if q.query_type == qtype),
            })

        write_csv(self.agg_path("oracle_routing.csv"), summary_rows)

        figures: List[str] = []
        path = self.figure_path("recall_by_query_type.png")
        if plot_query_type_grouped_bars(
            data=type_rows, metric="recall", out_path=path,
            title=f"Per-retriever recall@{_K} by query type ({corpus_name})",
        ):
            figures.append(str(path))

        findings: List[str] = []
        if best_overall is not None:
            gap = oracle_recall - single_best_recall
            findings.append(
                f"Best single retriever overall: **{best_overall}** "
                f"(recall@{_K}={single_best_recall:.3f}); oracle per-query routing "
                f"would reach {oracle_recall:.3f}, a gap of {gap:.3f} — this is the "
                "upper bound on adaptive query-type routing."
            )
        for qtype, retriever_key in best_per_type.items():
            findings.append(
                f"Best retriever for **{qtype}** queries: **{retriever_key}** "
                f"(recall@{_K}={results[retriever_key][qtype]:.3f})."
            )
        return {
            "figures": figures,
            "tables": [str(self.agg_path("recall_by_type_per_retriever.csv")),
                       str(self.agg_path("oracle_routing.csv"))],
            "findings": findings,
            "rows": type_rows,
        }
