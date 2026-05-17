"""Suite 2 — Top-k sensitivity & context-pollution onset.

Question: At which k does recall saturate? At which k does context pollution
become severe (irrelevant chunks crowd out gold)?
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS, TOPK_VALUES
from experiments.metrics import aggregate_retrieval_metrics, context_pollution_ratio
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_topk_sensitivity


_TARGET_CORPUS = "n1000"
_FALLBACK_CORPUS_ORDER = ["n1000", "n100", "n50", "n10"]


class TopKSensitivitySuite(Suite):
    key = "s02_topk_sensitivity"
    description = "How recall, precision and pollution evolve as top_k grows"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for top-k sensitivity suite.")

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
                    top_k=max(TOPK_VALUES),
                )
                raw_writer.write_many(records)

                gold = gold_lookup_chunks(queries)
                metrics = aggregate_retrieval_metrics(
                    [r.to_dict() for r in records],
                    gold,
                    k_values=TOPK_VALUES,
                )
                # Pollution per k.
                per_query: Dict[str, List[str]] = {}
                for r in records:
                    per_query.setdefault(r.query_id, []).append(r.chunk_id)

                for k in TOPK_VALUES:
                    poll = [
                        context_pollution_ratio(ids[:k], gold.get(qid, set()))
                        for qid, ids in per_query.items()
                    ]
                    avg_pollution = sum(poll) / len(poll) if poll else 0.0
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "top_k": k,
                        "recall": round(metrics[f"recall@{k}"], 6),
                        "precision": round(metrics[f"precision@{k}"], 6),
                        "ndcg": round(metrics[f"ndcg@{k}"], 6),
                        "hit": round(metrics[f"hit@{k}"], 6),
                        "context_pollution": round(avg_pollution, 6),
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("topk_sensitivity.csv"), agg_rows)

        figures: List[str] = []
        for metric in ("recall", "precision", "ndcg", "context_pollution"):
            path = self.figure_path(f"{metric}_vs_topk.png")
            if plot_topk_sensitivity(
                data=agg_rows,
                metric=metric,
                out_path=path,
                title=f"{metric} vs top_k ({corpus_name})",
            ):
                figures.append(str(path))

        findings = _interpret_topk(agg_rows)

        return {
            "figures": figures,
            "tables": [str(self.agg_path("topk_sensitivity.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_topk(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["No top-k rows produced."]

    findings: List[str] = []
    by_retriever: Dict[str, List[Dict]] = {}
    for r in rows:
        by_retriever.setdefault(r["retriever"], []).append(r)

    for retriever, group in by_retriever.items():
        group.sort(key=lambda r: r["top_k"])
        # Recall saturation: smallest k where increasing k adds < 0.02.
        saturation_k = None
        for i in range(1, len(group)):
            delta = group[i]["recall"] - group[i - 1]["recall"]
            if delta < 0.02:
                saturation_k = group[i - 1]["top_k"]
                break
        if saturation_k is not None:
            findings.append(
                f"**{retriever}** recall saturates around k={saturation_k}; "
                "further increases bring negligible gain."
            )
        # Pollution onset: smallest k where pollution > 0.5.
        onset = next((r["top_k"] for r in group if r["context_pollution"] > 0.5), None)
        if onset is not None:
            findings.append(
                f"**{retriever}** crosses 50% context pollution at k={onset}; "
                "generation faithfulness is at risk beyond this point."
            )

    return findings or ["Top-k behaviour appears smooth — no sharp saturation or pollution onset detected."]
