"""Suite 12 — Failure taxonomy.

Classifies each (query, retriever) outcome into one of:

* ``no_hit``           no gold chunk found in top_k
* ``low_rank``         gold chunk present only after rank 5
* ``polluted_context`` gold chunk present but > 50% of context is distractor
* ``generation_error`` the generator reported an error (only if enabled)
* ``ok``               none of the above

The distribution gives a compact picture of where each retriever loses.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_query_type_grouped_bars


_K = 10
_LOW_RANK_THRESHOLD = 5
_POLLUTION_THRESHOLD = 0.5
_FALLBACK_CORPUS_ORDER = ["n100", "n1000", "n50", "n10"]


class FailureTaxonomySuite(Suite):
    key = "s12_failure_taxonomy"
    description = "Distribution of retrieval failure modes per retriever"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for failure-taxonomy suite.")

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]
        gold = gold_lookup_chunks(queries)

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
                per_query: Dict[str, List] = {}
                for r in records:
                    per_query.setdefault(r.query_id, []).append(r)

                counts = {"no_hit": 0, "low_rank": 0,
                          "polluted_context": 0, "ok": 0}
                for query in queries:
                    rs = sorted(per_query.get(query.query_id, []), key=lambda r: r.rank)
                    gold_ids = gold.get(query.query_id, set())
                    if not rs:
                        counts["no_hit"] += 1
                        continue
                    chunk_ids = [r.chunk_id for r in rs]
                    hits = [i for i, cid in enumerate(chunk_ids) if cid in gold_ids]
                    if not hits:
                        counts["no_hit"] += 1
                        continue
                    pollution = 1.0 - (len(hits) / max(1, len(chunk_ids)))
                    if hits[0] >= _LOW_RANK_THRESHOLD:
                        counts["low_rank"] += 1
                    elif pollution > _POLLUTION_THRESHOLD:
                        counts["polluted_context"] += 1
                    else:
                        counts["ok"] += 1

                total = sum(counts.values()) or 1
                for mode, n in counts.items():
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "query_type": mode,  # plot reuses query_type column
                        "n": n,
                        "fraction": round(n / total, 6),
                        "n_queries": total,
                    })
        finally:
            raw_writer.close()

        write_csv(self.agg_path("failure_taxonomy.csv"), agg_rows)

        figures: List[str] = []
        path = self.figure_path("failure_distribution.png")
        if plot_query_type_grouped_bars(
            data=agg_rows, metric="fraction", out_path=path,
            title=f"Failure mode distribution ({corpus_name})",
        ):
            figures.append(str(path))

        findings = _interpret(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("failure_taxonomy.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No failure-taxonomy rows produced."]
    by_retr: Dict[str, Dict[str, float]] = {}
    for r in rows:
        by_retr.setdefault(r["retriever"], {})[r["query_type"]] = r["fraction"]

    for retriever, dist in by_retr.items():
        ok = dist.get("ok", 0.0)
        no_hit = dist.get("no_hit", 0.0)
        low_rank = dist.get("low_rank", 0.0)
        pollution = dist.get("polluted_context", 0.0)
        dominant = max(dist.items(), key=lambda kv: kv[1])
        findings.append(
            f"**{retriever}**: ok={ok:.2f}, no_hit={no_hit:.2f}, "
            f"low_rank={low_rank:.2f}, polluted={pollution:.2f}. "
            f"Dominant failure mode: **{dominant[0]}** ({dominant[1]:.2f})."
        )
    return findings
