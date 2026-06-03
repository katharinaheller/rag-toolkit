"""Suite 8 — Noise robustness.

Compares retrieval quality on noisy vs semantic-paraphrase queries. Identifies
which retrievers degrade least when query surface form is corrupted.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS
from rag.evaluation.metrics import aggregate_retrieval_metrics
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
_CLEAN_TYPE = "semantic_paraphrase"
_NOISY_TYPE = "noisy"


class NoiseRobustnessSuite(Suite):
    key = "s08_noise_robustness"
    description = "Retrieval degradation under noisy vs paraphrased queries"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for noise-robustness suite.")

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
                    self.logger.warning("Index build failed (%s/%s): %s",
                                        retriever.key, corpus_name, exc)
                    continue

                records = run_retriever_on_queries(
                    run_id=self.ctx.run_id, suite_key=self.key,
                    retriever=retriever, built_index=index,
                    corpus_name=corpus_name, queries=queries, top_k=_K,
                )
                raw_writer.write_many(records)

                gold = gold_lookup_chunks(queries)
                for qtype in (_CLEAN_TYPE, _NOISY_TYPE):
                    qids = {q.query_id for q in queries if q.query_type == qtype}
                    if not qids:
                        continue
                    subset_records = [r.to_dict() for r in records if r.query_id in qids]
                    subset_gold = {qid: g for qid, g in gold.items() if qid in qids}
                    if not subset_records:
                        continue
                    metrics = aggregate_retrieval_metrics(
                        subset_records, subset_gold, k_values=[_K],
                    )
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "query_type": qtype,
                        "n_queries": len(qids),
                        "recall": round(metrics[f"recall@{_K}"], 6),
                        "ndcg": round(metrics[f"ndcg@{_K}"], 6),
                        "mrr": round(metrics[f"mrr@{_K}"], 6),
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("noise_robustness.csv"), agg_rows)

        figures: List[str] = []
        for metric in ("recall", "ndcg", "mrr"):
            path = self.figure_path(f"{metric}_clean_vs_noisy.png")
            if plot_query_type_grouped_bars(
                data=agg_rows, metric=metric, out_path=path,
                title=f"{metric}@{_K}: paraphrase vs noisy",
            ):
                figures.append(str(path))

        findings = _interpret_noise(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("noise_robustness.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_noise(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No noise-robustness rows produced."]
    by_retr: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_retr.setdefault(r["retriever"], {})[r["query_type"]] = r

    deltas: List = []
    for retriever, types in by_retr.items():
        clean = types.get(_CLEAN_TYPE)
        noisy = types.get(_NOISY_TYPE)
        if clean is None or noisy is None:
            continue
        delta = clean["recall"] - noisy["recall"]
        deltas.append((retriever, delta, clean["recall"], noisy["recall"]))

    if not deltas:
        return ["Neither paraphrase nor noisy queries were available for comparison."]

    deltas.sort(key=lambda t: t[1])
    most_robust = deltas[0]
    least_robust = deltas[-1]
    findings.append(
        f"Most noise-robust retriever: **{most_robust[0]}** "
        f"(recall@{_K} drop {most_robust[1]:.3f}: {most_robust[2]:.3f} → {most_robust[3]:.3f})."
    )
    findings.append(
        f"Least noise-robust retriever: **{least_robust[0]}** "
        f"(recall@{_K} drop {least_robust[1]:.3f}: {least_robust[2]:.3f} → {least_robust[3]:.3f})."
    )
    return findings
