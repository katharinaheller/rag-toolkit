"""Markdown report builder.

Consumes the list of suite summary dicts produced by ``run_all_experiments``
and writes ``REPORT.md`` to ``outputs/reports/<run_id>/``. The report follows
the 15-section structure required by the thesis specification.

Each section embeds the relevant suite's figures (as relative links) and the
auto-generated findings strings — no hard-coded conclusions, only what the
data supports.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _rel(path: str, report_dir: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(report_dir.resolve()))
    except Exception:
        # Fall back to absolute path when the target lives outside the report tree.
        return str(Path(path).resolve())


def _section(report_dir: Path, title: str, summary: Optional[Dict[str, Any]]) -> List[str]:
    lines = [f"## {title}", ""]
    if not summary:
        lines.append("_Suite did not produce a summary in this run._")
        lines.append("")
        return lines
    desc = summary.get("description")
    if desc:
        lines.append(f"_{desc}_")
        lines.append("")
    status = summary.get("status", "ok")
    if status != "ok":
        lines.append(f"**Status:** `{status}`")
        if "error" in summary:
            lines.append(f"**Error:** `{summary['error']}`")
        lines.append("")
    findings = summary.get("findings", [])
    if findings:
        lines.append("**Findings:**")
        lines.append("")
        for f in findings:
            lines.append(f"- {f}")
        lines.append("")
    figures = summary.get("figures", [])
    if figures:
        lines.append("**Figures:**")
        lines.append("")
        for fig in figures:
            fig_rel = _rel(fig, report_dir)
            label = Path(fig).stem.replace("_", " ")
            lines.append(f"![{label}]({fig_rel})")
            lines.append("")
    tables = summary.get("tables", [])
    if tables:
        lines.append("**Tables:**")
        lines.append("")
        for tab in tables:
            tab_rel = _rel(tab, report_dir)
            lines.append(f"- [{Path(tab).name}]({tab_rel})")
        lines.append("")
    return lines


# Mapping from report section name to suite key.
_SECTION_TO_SUITE = [
    ("Retrieval Scaling Analysis", "s01_retriever_corpus_scaling"),
    ("Top-k Trade-off Analysis", "s02_topk_sensitivity"),
    ("Query-type Analysis", "s03_query_type_comparison"),
    ("Embedding Comparison", "s04_embedding_comparison"),
    ("Pareto / Latency-quality Analysis", "s05_latency_quality_pareto"),
    ("Retrieval Overlap Analysis", "s06_retrieval_overlap"),
    ("Hybrid Retrieval Analysis", "s07_hybrid_fusion_sweep"),
    ("Noise Robustness", "s08_noise_robustness"),
    ("Stability Analysis", "s09_stability"),
    ("Context Pollution Analysis", "s10_context_pollution"),
    ("Long-context Behaviour", "s11_long_context"),
    ("Failure Analysis", "s12_failure_taxonomy"),
    ("Chunk-relevance Decay", "s13_chunk_relevance_decay"),
    ("Resource / Performance Analysis", "s14_throughput_resources"),
    ("Query-conditioned Configuration Analysis", "s15_query_conditioned"),
]


def _executive_summary(summaries_by_key: Dict[str, Dict[str, Any]]) -> List[str]:
    """Pull the headline finding from each suite to make a thesis-grade abstract."""
    lines = ["## Executive Summary", ""]
    lines.append(
        "This report presents an end-to-end evaluation of a local RAG system "
        "covering retrieval, generation, scaling, and stability behaviour across "
        "four corpora (n10, n50, n100, n1000) and six retrievers (TF-IDF, BM25, "
        "dense Gemma, dense BGE, hybrid Gemma, hybrid BGE). All measurements "
        "below are produced by the suites listed in the table of contents; "
        "interpretation strings are auto-generated from the raw aggregated data."
    )
    lines.append("")
    bullets: List[str] = []
    for title, key in _SECTION_TO_SUITE:
        summary = summaries_by_key.get(key)
        if not summary:
            continue
        findings = summary.get("findings", [])
        if findings:
            bullets.append(f"- **{title}:** {findings[0]}")
    if bullets:
        lines.append("**Headline findings:**")
        lines.append("")
        lines.extend(bullets)
        lines.append("")
    return lines


def _practical_recommendation(summaries_by_key: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = ["## Practical Recommendation", ""]
    bullets: List[str] = []
    pareto = summaries_by_key.get("s05_latency_quality_pareto")
    if pareto and pareto.get("findings"):
        bullets.append(
            "Use the Pareto-front retrievers (see latency-quality analysis) for "
            "deployment; non-Pareto retrievers are dominated on either axis."
        )
    topk = summaries_by_key.get("s02_topk_sensitivity")
    if topk and topk.get("findings"):
        bullets.append(
            "Set operational top_k to the saturation point identified in the "
            "top-k sensitivity analysis to avoid wasted generation latency and "
            "downstream context pollution."
        )
    routing = summaries_by_key.get("s15_query_conditioned")
    if routing and routing.get("findings"):
        bullets.append(
            "If oracle vs single-retriever gap is significant, consider query-type "
            "classification before retrieval (see query-conditioned analysis)."
        )
    stab = summaries_by_key.get("s09_stability")
    if stab and stab.get("findings"):
        bullets.append(
            "For reproducible thesis results, prefer the retriever flagged as "
            "deterministic in the stability suite."
        )
    if not bullets:
        bullets.append("Insufficient data across suites to draw a global recommendation.")
    for b in bullets:
        lines.append(f"- {b}")
    lines.append("")
    return lines


def _reproducibility_notes(run_id: str, settings_dict: Dict[str, Any]) -> List[str]:
    lines = ["## Reproducibility Notes", ""]
    lines.append(f"- **Run ID:** `{run_id}`")
    lines.append(f"- **Seed:** `{settings_dict.get('seed')}`")
    lines.append(f"- **Queries per corpus:** `{settings_dict.get('n_queries_per_corpus')}`")
    lines.append(f"- **Output root:** `{settings_dict.get('output_root')}`")
    lines.append(f"- **Cache root:** `{settings_dict.get('cache_root')}`")
    lines.append(f"- **Generation enabled:** `{settings_dict.get('enable_generation')}`")
    lines.append(
        "- All synthetic queries are derived deterministically from corpus chunks "
        "with a fixed RNG seed; embeddings and indexes are cached by content "
        "digest so a re-run reuses prior artefacts."
    )
    lines.append(
        "- Generation is intentionally optional. If Ollama is unreachable, "
        "generation-dependent suites mark themselves as `skipped` rather than "
        "fabricating numbers."
    )
    lines.append("")
    return lines


def _future_work() -> List[str]:
    lines = ["## Future Work", ""]
    for item in [
        "Replace provenance-anchored synthetic queries with human-annotated "
        "gold relevance judgements to remove the recall ceiling imposed by "
        "single-chunk attribution.",
        "Add a learned-sparse retriever (BGE-M3 lexical weights) once the "
        "embedding pipeline emits sparse vectors at scale.",
        "Extend the failure taxonomy with LLM-grounded categories (factual "
        "vs stylistic hallucination) once an offline judge becomes available.",
        "Add a cross-encoder reranker stage and re-run the Pareto suite to see "
        "whether the latency/quality frontier moves outward.",
        "Pre-compute per-corpus index build times so the resource suite can "
        "report a complete cost picture (build + query latency + memory).",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _limitations() -> List[str]:
    lines = ["## Limitations", ""]
    for item in [
        "Synthetic queries are derived from the same chunks they target; "
        "metrics tend to overstate recall versus an externally curated set.",
        "Faithfulness is approximated by token overlap with the supplied "
        "context — it does not detect well-aligned hallucinations that recycle "
        "context vocabulary.",
        "Memory measurements depend on `psutil` and are process-level; in a "
        "multi-tenant JupyterHub container they include shared library RSS.",
        "Latency on a shared HPC node is influenced by co-tenant load; "
        "absolute numbers are not portable, only the relative ranking is.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def build_report(
    run_id: str,
    summaries: List[Dict[str, Any]],
    settings_dict: Dict[str, Any],
    report_dir: Path,
    dashboard_figure: Optional[Path] = None,
) -> Path:
    """Write ``REPORT.md`` and return its path."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "REPORT.md"

    summaries_by_key = {s.get("key", ""): s for s in summaries}

    lines: List[str] = []
    lines.append(f"# RAG Experiments Report — `{run_id}`")
    lines.append("")
    lines.append(f"_Generated: {datetime.datetime.utcnow().isoformat()}Z_")
    lines.append("")

    # Table of contents.
    lines.append("## Contents")
    lines.append("")
    section_titles = (
        ["Executive Summary"]
        + [t for t, _ in _SECTION_TO_SUITE]
        + ["Practical Recommendation", "Reproducibility Notes",
           "Limitations", "Future Work"]
    )
    for i, title in enumerate(section_titles, 1):
        anchor = title.lower().replace(" ", "-").replace("/", "")
        lines.append(f"{i}. [{title}](#{anchor})")
    lines.append("")

    lines.extend(_executive_summary(summaries_by_key))

    if dashboard_figure is not None:
        rel = _rel(str(dashboard_figure), report_dir)
        lines.append("**Run dashboard:**")
        lines.append("")
        lines.append(f"![dashboard]({rel})")
        lines.append("")

    for title, key in _SECTION_TO_SUITE:
        lines.extend(_section(report_dir, title, summaries_by_key.get(key)))

    lines.extend(_practical_recommendation(summaries_by_key))
    lines.extend(_reproducibility_notes(run_id, settings_dict))
    lines.extend(_limitations())
    lines.extend(_future_work())

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
