"""Suite 7 — Hybrid fusion weight sweep.

For each hybrid retriever, sweep ``dense_weight`` (sparse_weight = 1 - dense_weight)
and measure recall and nDCG. Identifies the regime in which fusion helps and
the optimum weight per retriever.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import (
    FUSION_WEIGHT_SWEEP,
    RETRIEVERS_BY_KEY,
)
from experiments.metrics import aggregate_retrieval_metrics
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_topk_sensitivity


_K = 10
_FALLBACK_CORPUS_ORDER = ["n1000", "n100", "n50", "n10"]
_HYBRID_KEYS = ("hybrid_bge", "hybrid_gemma")


class HybridFusionSweepSuite(Suite):
    key = "s07_hybrid_fusion_sweep"
    description = "Sweep hybrid fusion weights to find the dense/sparse balance"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for fusion-weight suite.")

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]

        agg_rows: List[Dict[str, Any]] = []
        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("retrieval_records.jsonl"))

        try:
            for hkey in _HYBRID_KEYS:
                if hkey not in RETRIEVERS_BY_KEY:
                    continue
                retriever = RETRIEVERS_BY_KEY[hkey]
                try:
                    index = get_or_build_index(retriever, chunks, corpus_name, index_cache)
                except Exception as exc:
                    self.logger.warning("Index build failed (%s/%s): %s",
                                        retriever.key, corpus_name, exc)
                    continue

                gold = gold_lookup_chunks(queries)
                for w in FUSION_WEIGHT_SWEEP:
                    records = run_retriever_on_queries(
                        run_id=self.ctx.run_id, suite_key=self.key,
                        retriever=retriever, built_index=index,
                        corpus_name=corpus_name, queries=queries, top_k=_K,
                        fusion_dense_weight=w, fusion_sparse_weight=1.0 - w,
                    )
                    # Tag records with fusion weight.
                    for r in records:
                        rec = r.to_dict()
                        rec["extras"] = {**rec.get("extras", {}), "dense_weight": w}
                        raw_writer.write(rec)

                    if not records:
                        continue
                    metrics = aggregate_retrieval_metrics(
                        [r.to_dict() for r in records], gold, k_values=[_K],
                    )
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "dense_weight": round(w, 3),
                        "sparse_weight": round(1.0 - w, 3),
                        "top_k": _K,
                        "recall": round(metrics[f"recall@{_K}"], 6),
                        "ndcg": round(metrics[f"ndcg@{_K}"], 6),
                        "mrr": round(metrics[f"mrr@{_K}"], 6),
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("fusion_sweep.csv"), agg_rows)

        # Re-shape rows for plot_topk_sensitivity: needs ``top_k`` column.
        # Re-use the topk sensitivity plotter by mapping dense_weight onto top_k axis.
        plot_rows = [{**r, "top_k": int(round(r["dense_weight"] * 100))} for r in agg_rows]

        figures: List[str] = []
        for metric in ("recall", "ndcg", "mrr"):
            path = self.figure_path(f"{metric}_vs_fusion_weight.png")
            if plot_topk_sensitivity(
                data=plot_rows, metric=metric, out_path=path,
                title=f"{metric}@{_K} vs dense_weight×100 ({corpus_name})",
            ):
                figures.append(str(path))

        findings = _interpret_fusion(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("fusion_sweep.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_fusion(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No fusion-sweep rows produced."]
    by_retriever: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_retriever.setdefault(r["retriever"], []).append(r)

    for retriever, group in by_retriever.items():
        group.sort(key=lambda r: r["dense_weight"])
        best = max(group, key=lambda r: r["recall"])
        pure_dense = next((r for r in group if r["dense_weight"] >= 0.999), None)
        pure_sparse = next((r for r in group if r["dense_weight"] <= 0.001), None)
        findings.append(
            f"**{retriever}**: best recall@{_K}={best['recall']:.3f} at "
            f"dense_weight={best['dense_weight']:.2f}."
        )
        if pure_dense and pure_sparse:
            mid_gain = best["recall"] - max(pure_dense["recall"], pure_sparse["recall"])
            if mid_gain > 0.01:
                findings.append(
                    f"**{retriever}** fusion beats both pure dense "
                    f"({pure_dense['recall']:.3f}) and pure sparse "
                    f"({pure_sparse['recall']:.3f}) — the mid-range gain is "
                    f"{mid_gain:.3f}, confirming complementary evidence."
                )
            else:
                findings.append(
                    f"**{retriever}** fusion gives at most {mid_gain:.3f} above the "
                    "best pure side — fusion adds little here."
                )
    return findings
