"""Suite 11 — Long-context behaviour.

Sweeps ``top_k`` to vary the context window size and measures generation
quality (context overlap, latency, prompt chars). If the generator is not
available the suite is skipped with a clear note.
"""

from __future__ import annotations

from typing import Any, Dict, List

from experiments.configs.default_matrix import RETRIEVERS_BY_KEY, TOPK_VALUES
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


_FALLBACK_CORPUS_ORDER = ["n100", "n1000", "n50", "n10"]
_PREFERRED_RETRIEVER = "hybrid_bge"


class LongContextSuite(Suite):
    key = "s11_long_context"
    description = "Generation quality and latency as context length grows"

    def _pick_corpus(self) -> str:
        for name in _FALLBACK_CORPUS_ORDER:
            if name in self.ctx.corpora_chunks and self.ctx.queries.get(name):
                return name
        raise RuntimeError("No usable corpus for long-context suite.")

    def _pick_retriever(self):
        if _PREFERRED_RETRIEVER in RETRIEVERS_BY_KEY:
            return RETRIEVERS_BY_KEY[_PREFERRED_RETRIEVER]
        return next(iter(RETRIEVERS_BY_KEY.values()))

    def run(self) -> Dict[str, Any]:
        if not self.ctx.generator.available:
            return {
                "figures": [], "tables": [],
                "findings": ["Generator unavailable — long-context suite skipped."],
                "rows": [], "status": "skipped",
            }

        corpus_name = self._pick_corpus()
        chunks = self.ctx.corpora_chunks[corpus_name]
        queries = self.ctx.queries[corpus_name]
        retriever = self._pick_retriever()
        chunk_by_id: Dict[str, Dict] = {c["id"]: c for c in chunks}

        index_cache: Dict = {}
        try:
            index = get_or_build_index(retriever, chunks, corpus_name, index_cache)
        except Exception as exc:
            return {"figures": [], "tables": [], "findings": [str(exc)],
                    "rows": [], "status": "failed"}

        records = run_retriever_on_queries(
            run_id=self.ctx.run_id, suite_key=self.key,
            retriever=retriever, built_index=index,
            corpus_name=corpus_name, queries=queries,
            top_k=max(TOPK_VALUES),
        )
        per_query: Dict[str, List[str]] = {}
        for r in records:
            per_query.setdefault(r.query_id, []).append(r.chunk_id)

        agg_rows: List[Dict[str, Any]] = []
        gen_writer = JsonlWriter(self.raw_path("generation_records.jsonl"))

        try:
            for k in TOPK_VALUES:
                overlaps: List[float] = []
                halluc: List[float] = []
                latencies: List[float] = []
                prompt_chars: List[int] = []
                context_chars: List[int] = []
                for query in queries:
                    chunk_ids = per_query.get(query.query_id, [])[:k]
                    ctx_chunks = [chunk_by_id[c]["text"] for c in chunk_ids if c in chunk_by_id]
                    if not ctx_chunks:
                        continue
                    gen = self.ctx.generator.generate(query.text, ctx_chunks)
                    o = context_overlap(gen.answer, ctx_chunks)
                    h = hallucination_score(gen.answer, ctx_chunks)
                    overlaps.append(o); halluc.append(h)
                    latencies.append(gen.latency_ms)
                    prompt_chars.append(gen.prompt_chars)
                    context_chars.append(gen.context_chars)

                    rec = GenerationRecord(
                        run_id=self.ctx.run_id, suite=self.key,
                        retriever=retriever.key, embedder=retriever.embedder_key,
                        corpus=corpus_name, top_k=k,
                        query_id=query.query_id, query_text=query.text,
                        query_type=query.query_type,
                        expected_answer=query.expected_answer,
                        generated_answer=gen.answer,
                        success=gen.error is None and bool(gen.answer),
                        latency_ms=gen.latency_ms,
                        prompt_chars=gen.prompt_chars,
                        context_chars=gen.context_chars,
                        error=gen.error,
                        extras={"context_overlap": o, "hallucination_score": h},
                    )
                    gen_writer.write(rec.to_dict())

                if overlaps:
                    agg_rows.append({
                        "retriever": retriever.key,
                        "corpus": corpus_name,
                        "top_k": k,
                        "mean_context_overlap": round(sum(overlaps) / len(overlaps), 6),
                        "mean_hallucination": round(sum(halluc) / len(halluc), 6),
                        "mean_gen_latency_ms": round(sum(latencies) / len(latencies), 3),
                        "mean_prompt_chars": round(sum(prompt_chars) / len(prompt_chars), 1),
                        "mean_context_chars": round(sum(context_chars) / len(context_chars), 1),
                        "n_queries": len(overlaps),
                    })
        finally:
            gen_writer.close()

        write_csv(self.agg_path("long_context.csv"), agg_rows)

        figures: List[str] = []
        for metric in ("mean_context_overlap", "mean_hallucination",
                       "mean_gen_latency_ms", "mean_context_chars"):
            path = self.figure_path(f"{metric}_vs_topk.png")
            if plot_topk_sensitivity(
                data=agg_rows, metric=metric, out_path=path,
                title=f"{metric} vs top_k",
            ):
                figures.append(str(path))

        findings = _interpret(agg_rows)
        return {
            "figures": figures,
            "tables": [str(self.agg_path("long_context.csv"))],
            "findings": findings,
            "rows": agg_rows,
        }


def _interpret(rows: List[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []
    if not rows:
        return ["No long-context rows produced."]
    rows_sorted = sorted(rows, key=lambda r: r["top_k"])
    smallest = rows_sorted[0]
    largest = rows_sorted[-1]
    findings.append(
        f"Context length grows from {smallest['mean_context_chars']:.0f} chars at "
        f"k={smallest['top_k']} to {largest['mean_context_chars']:.0f} chars at "
        f"k={largest['top_k']}; mean generation latency moves "
        f"{smallest['mean_gen_latency_ms']:.0f}ms → {largest['mean_gen_latency_ms']:.0f}ms."
    )
    overlap_delta = smallest["mean_context_overlap"] - largest["mean_context_overlap"]
    if overlap_delta > 0.05:
        findings.append(
            f"Context overlap drops by {overlap_delta:.3f} as context grows — "
            "longer contexts dilute the answer-grounding signal."
        )
    elif overlap_delta < -0.05:
        findings.append(
            f"Context overlap rises by {-overlap_delta:.3f} with more context — "
            "the generator benefits from the extra evidence."
        )
    return findings
