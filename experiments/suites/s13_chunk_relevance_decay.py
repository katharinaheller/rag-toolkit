"""Suite 13 — Chunk-relevance decay by rank.

For each rank position (0 .. max_k − 1) we compute the fraction of queries
where that rank contained a gold chunk. The expected shape is monotonic
decline; sharp drops indicate retrievers that concentrate signal near the top.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS, TOPK_VALUES
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_rank_decay


_FALLBACK_CORPUS_ORDER = ["n100", "n1000", "n50", "n10"]


class ChunkRelevanceDecaySuite(Suite):
    key = "s13_chunk_relevance_decay"
    description = "Per-rank hit rate to expose where retrieval signal concentrates"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for chunk-relevance-decay suite.")

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]
        gold = gold_lookup_chunks(queries)
        max_k = max(TOPK_VALUES)

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
                    corpus_name=corpus_name, queries=queries, top_k=max_k,
                )
                raw_writer.write_many(records)
                per_query: Dict[str, List] = {}
                for r in records:
                    per_query.setdefault(r.query_id, []).append(r)

                rank_hits = [0] * max_k
                n_queries = 0
                for q in queries:
                    rs = sorted(per_query.get(q.query_id, []), key=lambda r: r.rank)
                    if not rs:
                        continue
                    n_queries += 1
                    gold_ids = gold.get(q.query_id, set())
                    for r in rs[:max_k]:
                        if r.chunk_id in gold_ids:
                            rank_hits[r.rank] += 1
                if n_queries == 0:
                    continue
                for rank in range(max_k):
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "rank": rank,
                        "hit_rate": round(rank_hits[rank] / n_queries, 6),
                        "n_queries": n_queries,
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("rank_decay.csv"), agg_rows)

        figures: List[str] = []
        path = self.figure_path("hit_rate_by_rank.png")
        if plot_rank_decay(
            data=agg_rows, out_path=path,
            title=f"Hit rate by rank ({corpus_name})",
        ):
            figures.append(str(path))

        findings = _interpret(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("rank_decay.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No rank-decay rows produced."]
    by_retr: Dict[str, List[Dict]] = {}
    for r in rows:
        by_retr.setdefault(r["retriever"], []).append(r)

    for retriever, group in by_retr.items():
        group.sort(key=lambda r: r["rank"])
        rank0 = group[0]["hit_rate"] if group else 0.0
        # Half-life: smallest rank where hit_rate <= rank0/2.
        half = next((r["rank"] for r in group if r["hit_rate"] <= rank0 / 2.0 and rank0 > 0), None)
        if half is not None:
            findings.append(
                f"**{retriever}**: rank-0 hit rate {rank0:.3f}; halves by rank "
                f"{half}. Signal is concentrated near the top."
            )
        else:
            findings.append(
                f"**{retriever}**: rank-0 hit rate {rank0:.3f}; no clear half-life "
                "within the measured top-k — relevance is distributed."
            )
    return findings
