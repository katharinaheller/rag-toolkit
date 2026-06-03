"""Suite 6 — Retrieval overlap analysis.

Computes pairwise top-k Jaccard overlap between retrievers on the largest
available corpus. Low overlap pairs are complementary — the empirical
justification for hybrid retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS
from rag.evaluation.metrics import pairwise_overlap_matrix
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_heatmap


_K = 10
_FALLBACK_CORPUS_ORDER = ["n1000", "n100", "n50", "n10"]


class RetrievalOverlapSuite(Suite):
    key = "s06_retrieval_overlap"
    description = "Pairwise retrieval overlap to identify complementary retrievers"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for retrieval-overlap suite.")

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]

        index_cache: Dict = {}
        raw_writer = JsonlWriter(self.raw_path("retrieval_records.jsonl"))

        # Collect top-k chunk ids per (retriever, query).
        retrieved: Dict[str, Dict[str, List[str]]] = {}

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
                bucket: Dict[str, List[str]] = {}
                for r in records:
                    bucket.setdefault(r.query_id, []).append(r.chunk_id)
                retrieved[retriever.key] = bucket
        finally:
            raw_writer.close()

        matrix = pairwise_overlap_matrix(retrieved, k=_K)

        # Flatten matrix to CSV.
        agg_rows: List[Dict[str, Any]] = []
        for a, row in matrix.items():
            for b, v in row.items():
                agg_rows.append({"retriever_a": a, "retriever_b": b, "overlap_at_k": round(v, 6)})
        write_csv(self.agg_path("overlap_matrix.csv"), agg_rows)

        figures: List[str] = []
        fig_path = self.figure_path("overlap_heatmap.png")
        if plot_heatmap(
            matrix=matrix, out_path=fig_path,
            title=f"Top-{_K} retrieval overlap on {corpus_name}",
        ):
            figures.append(str(fig_path))

        findings = _interpret_overlap(matrix)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("overlap_matrix.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret_overlap(matrix: Dict[str, Dict[str, float]]) -> List[str]:
    findings: List[str] = []
    keys = sorted(matrix.keys())
    pairs: List = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            pairs.append((a, b, matrix[a][b]))
    if not pairs:
        return ["Insufficient data for overlap analysis."]

    pairs.sort(key=lambda p: p[2])
    most_complementary = pairs[0]
    most_redundant = pairs[-1]
    findings.append(
        f"Most complementary retrievers: **{most_complementary[0]}** and "
        f"**{most_complementary[1]}** (Jaccard@{_K}={most_complementary[2]:.3f}); "
        f"low overlap motivates fusion."
    )
    findings.append(
        f"Most redundant retrievers: **{most_redundant[0]}** and "
        f"**{most_redundant[1]}** (Jaccard@{_K}={most_redundant[2]:.3f}); "
        f"fusion of this pair adds little new evidence."
    )
    return findings
