"""Suite 9 — Stability / repeated runs.

Re-runs each retrieval ``STABILITY_REPEATS`` times for the same queries and
reports rank stability and bootstrap CIs around recall@k. The same query is
re-sent so any variation reveals non-determinism in the retrieval stack
(scoring, tie-breaking, fusion).
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS, STABILITY_REPEATS
from rag.evaluation.metrics import (
    aggregate_retrieval_metrics,
    bootstrap_ci,
    coefficient_of_variation,
    rank_stability,
)
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_stability_bars


_K = 10
_FALLBACK_CORPUS_ORDER = ["n100", "n1000", "n50", "n10"]


class StabilitySuite(Suite):
    key = "s09_stability"
    description = "Variance and rank stability across repeated retrieval runs"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for stability suite.")

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
                    repeats=STABILITY_REPEATS,
                )
                raw_writer.write_many(records)
                if not records:
                    continue

                gold = gold_lookup_chunks(queries)
                # Per-repeat recall.
                per_repeat_recall: List[float] = []
                rank_lists_per_query: Dict[str, List[List[str]]] = {}
                for repeat in range(STABILITY_REPEATS):
                    subset = [r.to_dict() for r in records if r.repeat_index == repeat]
                    if not subset:
                        continue
                    metrics = aggregate_retrieval_metrics(subset, gold, k_values=[_K])
                    per_repeat_recall.append(metrics[f"recall@{_K}"])
                    # Collect rank lists for stability computation.
                    for r in records:
                        if r.repeat_index == repeat:
                            rank_lists_per_query.setdefault(r.query_id, [[] for _ in range(STABILITY_REPEATS)])
                            rank_lists_per_query[r.query_id][repeat].append(r.chunk_id)

                # Aggregate metrics.
                cv = coefficient_of_variation(per_repeat_recall)
                _, lo, hi = (
                    bootstrap_ci(per_repeat_recall, n_resamples=200)
                    if per_repeat_recall else (0.0, 0.0, 0.0)
                )

                # Mean rank stability (Kendall-style) across repeats.
                stabilities = []
                for qid, lists in rank_lists_per_query.items():
                    valid = [lst for lst in lists if lst]
                    if len(valid) >= 2:
                        stabilities.append(rank_stability(valid, k=_K))
                mean_stability = sum(stabilities) / len(stabilities) if stabilities else 0.0

                agg_rows.append({
                    "retriever": retriever.key,
                    "corpus": corpus_name,
                    "repeats": STABILITY_REPEATS,
                    "mean_recall": round(sum(per_repeat_recall) / max(1, len(per_repeat_recall)), 6),
                    "mean_recall_ci_lo": round(lo, 6),
                    "mean_recall_ci_hi": round(hi, 6),
                    "cv_recall": round(cv, 6),
                    "rank_stability": round(mean_stability, 6),
                })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("stability.csv"), agg_rows)

        figures: List[str] = []
        fig = self.figure_path("recall_stability_bars.png")
        if plot_stability_bars(
            data=agg_rows,
            metric="mean_recall",
            out_path=fig,
            title=f"recall@{_K} ± bootstrap CI ({corpus_name})",
        ):
            figures.append(str(fig))

        findings = _interpret_stability(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("stability.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_stability(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No stability rows produced."]
    rows_sorted = sorted(rows, key=lambda r: r["cv_recall"])
    most_stable = rows_sorted[0]
    least_stable = rows_sorted[-1]
    findings.append(
        f"Most stable: **{most_stable['retriever']}** "
        f"(CV(recall)={most_stable['cv_recall']:.4f}, rank_stability="
        f"{most_stable['rank_stability']:.3f})."
    )
    findings.append(
        f"Least stable: **{least_stable['retriever']}** "
        f"(CV(recall)={least_stable['cv_recall']:.4f}, rank_stability="
        f"{least_stable['rank_stability']:.3f})."
    )
    for r in rows_sorted:
        if r["cv_recall"] < 1e-6:
            findings.append(
                f"**{r['retriever']}** is fully deterministic across {r['repeats']} repeats."
            )
            break
    return findings
