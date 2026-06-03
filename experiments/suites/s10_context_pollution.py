"""Suite 10 — Controlled context-pollution injection.

For each query we keep ``top_k`` chunks but progressively replace gold chunks
with random distractors at increasing ratios. If the generator is available we
measure context-grounded faithfulness; otherwise we report only retrieval-side
pollution metrics and clearly skip generation rows.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS_BY_KEY
from experiments.configs.settings import SETTINGS
from experiments.core.types import GenerationRecord
from rag.evaluation.metrics import context_overlap, hallucination_score
from experiments.storage import JsonlWriter, write_csv
from experiments.suites._shared import (
    gold_lookup_chunks,
    get_or_build_index,
    run_retriever_on_queries,
)
from experiments.suites.base import Suite
from experiments.visualisation.plots import plot_topk_sensitivity


_K = 10
_POLLUTION_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
_PREFERRED_RETRIEVER = "hybrid_bge"
_FALLBACK_CORPUS_ORDER = ["n100", "n1000", "n50", "n10"]


class ContextPollutionSuite(Suite):
    key = "s10_context_pollution"
    description = "Faithfulness vs controlled context pollution ratio"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for context-pollution suite.")

    def _pick_retriever(self):
        if _PREFERRED_RETRIEVER in RETRIEVERS_BY_KEY:
            return RETRIEVERS_BY_KEY[_PREFERRED_RETRIEVER]
        return next(iter(RETRIEVERS_BY_KEY.values()))

    def run(self) -> Dict[str, Any]:
        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]
        retriever = self._pick_retriever()

        if not self.ctx.generator.available:
            return self._retrieval_only(corpus_name, chunks, queries, retriever)

        # Build chunk_id → text map for context construction.
        chunk_by_id: Dict[str, Dict] = {c["id"]: c for c in chunks}
        index_cache: Dict = {}
        try:
            index = get_or_build_index(retriever, chunks, corpus_name, index_cache)
        except Exception as exc:
            self.logger.warning("Index build failed for %s on %s: %s",
                                retriever.key, corpus_name, exc)
            return {"figures": [], "tables": [], "findings": [str(exc)], "rows": []}

        records = run_retriever_on_queries(
            run_id=self.ctx.run_id, suite_key=self.key,
            retriever=retriever, built_index=index,
            corpus_name=corpus_name, queries=queries, top_k=_K,
        )
        gold = gold_lookup_chunks(queries)

        # Per-query retrieved chunk lists.
        per_query: Dict[str, List[str]] = {}
        for r in records:
            per_query.setdefault(r.query_id, []).append(r.chunk_id)

        rng = random.Random(SETTINGS.seed + 13)
        distractor_ids = [c["id"] for c in chunks]

        agg_rows: List[Dict[str, Any]] = []
        gen_writer = JsonlWriter(self.raw_path("generation_records.jsonl"))

        try:
            for query in queries:
                retrieved = per_query.get(query.query_id, [])[:_K]
                if not retrieved:
                    continue
                gold_set = gold.get(query.query_id, set())

                for ratio in _POLLUTION_RATIOS:
                    n_to_replace = int(round(ratio * _K))
                    # Replace n_to_replace gold-with-distractors deterministically.
                    polluted = list(retrieved)
                    gold_positions = [i for i, cid in enumerate(polluted) if cid in gold_set]
                    rng.shuffle(gold_positions)
                    for pos in gold_positions[:n_to_replace]:
                        # Pick a distractor not already in polluted.
                        attempts = 0
                        while attempts < 50:
                            cand = rng.choice(distractor_ids)
                            if cand not in polluted and cand not in gold_set:
                                polluted[pos] = cand
                                break
                            attempts += 1

                    context_chunks = [
                        chunk_by_id[cid]["text"] for cid in polluted if cid in chunk_by_id
                    ]
                    gen = self.ctx.generator.generate(query.text, context_chunks)
                    overlap = context_overlap(gen.answer, context_chunks)
                    halluc = hallucination_score(gen.answer, context_chunks)

                    rec = GenerationRecord(
                        run_id=self.ctx.run_id, suite=self.key,
                        retriever=retriever.key, embedder=retriever.embedder_key,
                        corpus=corpus_name, top_k=_K,
                        query_id=query.query_id, query_text=query.text,
                        query_type=query.query_type,
                        expected_answer=query.expected_answer,
                        generated_answer=gen.answer,
                        success=gen.error is None and bool(gen.answer),
                        latency_ms=gen.latency_ms,
                        prompt_chars=gen.prompt_chars,
                        context_chars=gen.context_chars,
                        error=gen.error,
                        extras={"pollution_ratio": ratio,
                                "context_overlap": overlap,
                                "hallucination_score": halluc},
                    )
                    gen_writer.write(rec.to_dict())

                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "top_k": _K,
                        "pollution_ratio": ratio,
                        "query_id": query.query_id,
                        "context_overlap": round(overlap, 6),
                        "hallucination_score": round(halluc, 6),
                        "success": rec.success,
                    })
        finally:
            gen_writer.close()

        # Aggregate per ratio.
        summary_rows: List[Dict[str, Any]] = []
        for ratio in _POLLUTION_RATIOS:
            subset = [r for r in agg_rows if r["pollution_ratio"] == ratio]
            if not subset:
                continue
            summary_rows.append({
                "retriever": retriever.key,
                "corpus": corpus_name,
                "top_k": int(round(ratio * 100)),
                "pollution_ratio": ratio,
                "mean_context_overlap": round(
                    sum(r["context_overlap"] for r in subset) / len(subset), 6),
                "mean_hallucination": round(
                    sum(r["hallucination_score"] for r in subset) / len(subset), 6),
                "n_queries": len(subset),
            })

        write_csv(self.agg_path("pollution_effect.csv"), summary_rows)
        write_csv(self.agg_path("per_query_pollution.csv"), agg_rows)

        figures: List[str] = []
        for metric in ("mean_context_overlap", "mean_hallucination"):
            path = self.figure_path(f"{metric}_vs_pollution.png")
            if plot_topk_sensitivity(
                data=summary_rows, metric=metric, out_path=path,
                title=f"{metric} vs pollution ratio×100",
            ):
                figures.append(str(path))

        findings = _interpret(summary_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("pollution_effect.csv")),
                       str(self.agg_path("per_query_pollution.csv"))],
            "findings": findings,
            "rows": summary_rows,
        }

    def _retrieval_only(self, corpus_name, chunks, queries, retriever) -> Dict[str, Any]:
        return {
            "figures": [],
            "tables": [],
            "findings": [
                "Generator unavailable (Ollama not reachable). "
                "Suite skipped — re-run with EXPERIMENTS_ENABLE_GENERATION=1 and a "
                "running Ollama instance to obtain faithfulness measurements."
            ],
            "rows": [],
            "status": "skipped",
        }


def _interpret(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No pollution rows produced."]
    rows_sorted = sorted(rows, key=lambda r: r["pollution_ratio"])
    baseline = rows_sorted[0]
    worst = rows_sorted[-1]
    drop = baseline["mean_context_overlap"] - worst["mean_context_overlap"]
    findings.append(
        f"Context overlap drops by {drop:.3f} as pollution ratio rises from "
        f"{baseline['pollution_ratio']:.2f} to {worst['pollution_ratio']:.2f} "
        f"({baseline['mean_context_overlap']:.3f} → {worst['mean_context_overlap']:.3f})."
    )
    # First sharp inflection.
    for i in range(1, len(rows_sorted)):
        delta = rows_sorted[i - 1]["mean_context_overlap"] - rows_sorted[i]["mean_context_overlap"]
        if delta > 0.1:
            findings.append(
                f"Sharp faithfulness drop ({delta:.3f}) when pollution moves "
                f"{rows_sorted[i - 1]['pollution_ratio']:.2f} → "
                f"{rows_sorted[i]['pollution_ratio']:.2f}; this is the practical "
                "tolerance limit for distractor chunks."
            )
            break
    return findings
