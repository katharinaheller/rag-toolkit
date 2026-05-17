"""Suite 1 — Retriever × Corpus scaling matrix.

Question: How does each retriever's recall@k change as the corpus grows from
n10 → n1000? Are there crossovers where a retriever overtakes another?
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import (
    CORPORA,
    RETRIEVERS,
    TOPK_VALUES,
)
from experiments.metrics import aggregate_retrieval_metrics
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_retriever_corpus_scaling


_FOCUS_K = 10


class RetrieverCorpusScalingSuite(Suite):
    key = "s01_retriever_corpus_scaling"
    description = "How retrieval quality scales from n10 to n1000 per retriever"

    def run(self) -> Dict[str, Any]:
        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("retrieval_records.jsonl"))
        agg_rows: List[Dict[str, Any]] = []

        try:
            for corpus_name in CORPORA:
                chunks = self.ctx.corpora_chunks.get(corpus_name)
                queries = self.ctx.queries.get(corpus_name, [])
                if not chunks or not queries:
                    self.logger.warning(
                        "Skipping %s — chunks=%d queries=%d",
                        corpus_name, len(chunks or []), len(queries),
                    )
                    continue

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
                    self.logger.info(
                        "Retrieved %s × %s: %d records",
                        retriever.key, corpus_name, len(records),
                    )

                    metrics = aggregate_retrieval_metrics(
                        [r.to_dict() for r in records],
                        gold_lookup_chunks(queries),
                        k_values=TOPK_VALUES,
                    )
                    row: Dict[str, Any] = {
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "corpus_size": len(chunks),
                        "n_queries": len(queries),
                    }
                    row.update({k: round(v, 6) for k, v in metrics.items()})
                    agg_rows.append(row)
        finally:
            raw_writer.close()

        write_csv(self.agg_path("scaling_metrics.csv"), agg_rows)

        # Plot for each metric and each k.
        figures: List[str] = []
        for k in TOPK_VALUES:
            fig_path = self.figure_path(f"recall_at_{k}_vs_corpus_size.png")
            saved = plot_retriever_corpus_scaling(
                data=[{**r, "recall@k": r[f"recall@{k}"]} for r in agg_rows],
                metric=f"recall@{k}",
                out_path=fig_path,
                title=f"recall@{k} vs corpus size",
            )
            if saved:
                figures.append(str(saved))

        fig_ndcg = self.figure_path(f"ndcg_at_{_FOCUS_K}_vs_corpus_size.png")
        if plot_retriever_corpus_scaling(
            data=agg_rows,
            metric=f"ndcg@{_FOCUS_K}",
            out_path=fig_ndcg,
            title=f"nDCG@{_FOCUS_K} vs corpus size",
        ):
            figures.append(str(fig_ndcg))

        findings = _interpret_scaling(agg_rows)

        return {
            "figures": figures,
            "tables": [str(self.agg_path("scaling_metrics.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_scaling(rows: List[Dict[str, Any]]) -> List[str]:
    """Detect crossover behaviour and the best retriever per corpus."""
    findings: List[str] = []
    corpora = sorted({r["corpus_size"] for r in rows})
    if not rows:
        return ["No scaling rows produced — investigate index/query failures."]

    # Best retriever per corpus.
    for c in corpora:
        subset = [r for r in rows if r["corpus_size"] == c]
        subset.sort(key=lambda r: r.get(f"recall@{_FOCUS_K}", 0.0), reverse=True)
        if subset:
            best = subset[0]
            findings.append(
                f"On n{best['corpus_size']} the best retriever by recall@{_FOCUS_K} "
                f"is **{best['retriever']}** ({best[f'recall@{_FOCUS_K}']:.3f}). "
                f"Second-best gap: "
                f"{best[f'recall@{_FOCUS_K}'] - subset[1][f'recall@{_FOCUS_K}']:.3f}."
                if len(subset) > 1 else
                f"On n{best['corpus_size']} only one retriever produced results: {best['retriever']}."
            )

    # Crossover detection.
    by_retr: Dict[str, List[Dict]] = {}
    for r in rows:
        by_retr.setdefault(r["retriever"], []).append(r)
    for r_list in by_retr.values():
        r_list.sort(key=lambda r: r["corpus_size"])

    retriever_names = sorted(by_retr.keys())
    for i in range(len(retriever_names)):
        for j in range(i + 1, len(retriever_names)):
            a, b = retriever_names[i], retriever_names[j]
            la, lb = by_retr[a], by_retr[b]
            for idx in range(min(len(la), len(lb)) - 1):
                sa1, sb1 = la[idx][f"recall@{_FOCUS_K}"], lb[idx][f"recall@{_FOCUS_K}"]
                sa2, sb2 = la[idx + 1][f"recall@{_FOCUS_K}"], lb[idx + 1][f"recall@{_FOCUS_K}"]
                if (sa1 - sb1) * (sa2 - sb2) < 0:
                    findings.append(
                        f"Crossover detected between **{a}** and **{b}** between "
                        f"n{la[idx]['corpus_size']} and n{la[idx + 1]['corpus_size']} — "
                        "small-corpus winner is overtaken at larger scale."
                    )
    return findings or ["No crossover detected; ranking stable across corpus sizes."]
