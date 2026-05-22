"""Markdown report builder.

Two modes:

1. **In-pipeline** — ``run_all_experiments`` calls :func:`build_report` with the
   live list of suite summary dicts.
2. **Standalone** — ``python experiments/reports/report_builder.py`` rebuilds the
   report for the most recent run by reading the per-suite
   ``_suite_summary.json`` files persisted under
   ``outputs/aggregated/<run_id>/<suite>/``. It also writes the cross-run
   aggregated report. No experiments are re-run.

The section list now covers the original 15 research suites plus the six
GPU-aware benchmark / HPC resource suites (s16-s21). Unknown suite keys are
appended automatically so future suites still appear without code changes.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _rel(path: str, report_dir: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(report_dir.resolve()))
    except Exception:
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
    # GPU-aware benchmark / HPC resource suites.
    ("GPU vs CPU Embedding Benchmark", "s16_gpu_embedding_benchmark"),
    ("GPU vs CPU Generation Benchmark", "s17_gpu_generation_benchmark"),
    ("Batch-size Throughput Scaling", "s18_batchsize_scaling"),
    ("Cold Start vs Warm Steady-state", "s19_cold_warm"),
    ("HPC Resource Timelines", "s20_resource_timeline"),
    ("Concurrency Scaling", "s21_concurrency_scaling"),
]

_KNOWN_KEYS = {key for _, key in _SECTION_TO_SUITE}


def _executive_summary(summaries_by_key: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = ["## Executive Summary", ""]
    lines.append(
        "This report presents an end-to-end evaluation of a local RAG system "
        "covering retrieval, generation, scaling and stability behaviour, "
        "extended with GPU-aware benchmarks (CPU vs GPU embedding/generation, "
        "batch-size and concurrency scaling, cold vs warm start) and HPC-style "
        "resource timelines. All measurements are produced by the suites listed "
        "in the table of contents; interpretation strings are auto-generated "
        "from the raw aggregated data."
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


def _hardware_section(summaries_by_key: Dict[str, Dict[str, Any]]) -> List[str]:
    """Surface the captured hardware/CUDA metadata if any benchmark recorded it."""
    lines = ["## Hardware & Environment", ""]
    bench_keys = [k for k in summaries_by_key if k.startswith(("s16", "s17", "s18", "s19", "s20", "s21"))]
    hw_lines: List[str] = []
    for k in bench_keys:
        summary = summaries_by_key.get(k, {})
        for f in summary.get("findings", []):
            if any(tok in f for tok in ("CUDA", "Host CPU", "torch")):
                if f not in hw_lines:
                    hw_lines.append(f)
    if not hw_lines:
        hw_lines.append("No hardware metadata captured in this run.")
    for h in hw_lines:
        lines.append(f"- {h}")
    lines.append("")
    return lines


def _practical_recommendation(summaries_by_key: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = ["## Practical Recommendation", ""]
    bullets: List[str] = []
    if summaries_by_key.get("s05_latency_quality_pareto", {}).get("findings"):
        bullets.append(
            "Use the Pareto-front retrievers (see latency-quality analysis) for "
            "deployment; non-Pareto retrievers are dominated on either axis."
        )
    if summaries_by_key.get("s02_topk_sensitivity", {}).get("findings"):
        bullets.append(
            "Set operational top_k to the saturation point identified in the "
            "top-k sensitivity analysis to avoid wasted generation latency."
        )
    if summaries_by_key.get("s16_gpu_embedding_benchmark", {}).get("findings"):
        bullets.append(
            "Provision GPU for embedding only where the measured speedup "
            "(s16) justifies the VRAM cost; otherwise the CPU path is adequate."
        )
    if summaries_by_key.get("s18_batchsize_scaling", {}).get("findings"):
        bullets.append(
            "Pick the embedding batch size at the throughput saturation knee "
            "(s18) rather than the largest batch, to bound VRAM."
        )
    if summaries_by_key.get("s21_concurrency_scaling", {}).get("findings"):
        bullets.append(
            "Size the retrieval worker pool to the concurrency level where "
            "throughput stops scaling (s21); beyond it latency grows for no gain."
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
    lines.append(f"- **GPU benchmarks enabled:** `{settings_dict.get('enable_gpu_benchmarks')}`")
    lines.append(
        "- Benchmarks set torch/numpy/random seeds per variant; GPU variants "
        "reset the per-device VRAM peak counter so reported peaks are isolated."
    )
    lines.append(
        "- GPU variants are skipped (not failed) when CUDA is unavailable, so "
        "the same pipeline reproduces on CPU-only and GPU hosts."
    )
    lines.append("")
    return lines


def _limitations() -> List[str]:
    lines = ["## Limitations", ""]
    for item in [
        "Synthetic queries are derived from the same chunks they target; "
        "metrics tend to overstate recall versus an externally curated set.",
        "Generation device placement is server-side (Ollama); the CPU/GPU "
        "labels in s17 document the configured backend, while the local monitor "
        "still captures host GPU usage when the server is co-located.",
        "Latency on a shared host is influenced by co-tenant load; absolute "
        "numbers are not portable, only the relative ranking is.",
        "VRAM peaks reflect the torch allocator's view; framework-external "
        "allocations (e.g. cuBLAS workspaces) may not be fully attributed.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _future_work() -> List[str]:
    lines = ["## Future Work", ""]
    for item in [
        "Add multi-GPU sharding benchmarks once a multi-GPU host is available.",
        "Measure mixed-precision (fp16/bf16) vs fp32 throughput and quality "
        "trade-offs directly in the batch-size suite.",
        "Add a cross-encoder reranker stage and re-run the Pareto + resource "
        "suites to see whether the latency/quality frontier moves outward.",
        "Replace synthetic queries with human-annotated gold judgements.",
    ]:
        lines.append(f"- {item}")
    lines.append("")
    return lines


def _extra_sections(
    report_dir: Path, summaries_by_key: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Render any suite summaries whose key is not in the static section map."""
    lines: List[str] = []
    for key, summary in summaries_by_key.items():
        if key in _KNOWN_KEYS or not key:
            continue
        title = summary.get("description") or key
        lines.extend(_section(report_dir, f"{title} (`{key}`)", summary))
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

    lines.append("## Contents")
    lines.append("")
    section_titles = (
        ["Executive Summary", "Hardware & Environment"]
        + [t for t, _ in _SECTION_TO_SUITE]
        + ["Practical Recommendation", "Reproducibility Notes",
           "Limitations", "Future Work"]
    )
    for i, title in enumerate(section_titles, 1):
        anchor = title.lower().replace(" ", "-").replace("/", "").replace("&", "")
        anchor = anchor.replace("--", "-")
        lines.append(f"{i}. [{title}](#{anchor})")
    lines.append("")

    lines.extend(_executive_summary(summaries_by_key))
    lines.extend(_hardware_section(summaries_by_key))

    if dashboard_figure is not None:
        rel = _rel(str(dashboard_figure), report_dir)
        lines.append("**Run dashboard:**")
        lines.append("")
        lines.append(f"![dashboard]({rel})")
        lines.append("")

    for title, key in _SECTION_TO_SUITE:
        lines.extend(_section(report_dir, title, summaries_by_key.get(key)))

    lines.extend(_extra_sections(report_dir, summaries_by_key))
    lines.extend(_practical_recommendation(summaries_by_key))
    lines.extend(_reproducibility_notes(run_id, settings_dict))
    lines.extend(_limitations())
    lines.extend(_future_work())

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ── Standalone offline rebuild ──────────────────────────────────────────────
def _discover_runs(output_root: Path) -> List[str]:
    agg = output_root / "aggregated"
    if not agg.exists():
        return []
    runs = [p.name for p in agg.iterdir() if p.is_dir()]
    runs.sort()
    return runs


def _load_summaries_for_run(output_root: Path, run_id: str) -> List[Dict[str, Any]]:
    agg = output_root / "aggregated" / run_id
    summaries: List[Dict[str, Any]] = []
    if not agg.exists():
        return summaries
    for suite_dir in sorted(agg.iterdir()):
        if not suite_dir.is_dir():
            continue
        summary_file = suite_dir / "_suite_summary.json"
        if summary_file.exists():
            try:
                summaries.append(json.loads(summary_file.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning("Failed to read %s: %s", summary_file, exc)
    return summaries


def rebuild_latest_report(output_root: Optional[Path] = None) -> Optional[Path]:
    """Rebuild ``REPORT.md`` for the most recent run from persisted summaries."""
    from experiments.configs.settings import SETTINGS

    root = Path(output_root) if output_root else SETTINGS.output_root
    runs = _discover_runs(root)
    if not runs:
        logger.error("No runs found under %s/aggregated.", root)
        return None
    run_id = runs[-1]
    summaries = _load_summaries_for_run(root, run_id)
    if not summaries:
        logger.error("No suite summaries found for run %s.", run_id)
        return None

    report_dir = root / "reports" / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the dashboard if it was generated earlier.
    dashboard = report_dir / "dashboard.png"
    dashboard_path = dashboard if dashboard.exists() else None

    import dataclasses
    settings_dict = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in dataclasses.asdict(SETTINGS).items()
    }
    report_path = build_report(
        run_id=run_id,
        summaries=summaries,
        settings_dict=settings_dict,
        report_dir=report_dir,
        dashboard_figure=dashboard_path,
    )
    logger.info("Rebuilt report for run %s: %s", run_id, report_path)
    return report_path


def main() -> int:
    """Entry point for ``python experiments/reports/report_builder.py``.

    Rebuilds the latest run's Markdown report and the cross-run aggregated
    report, both purely from persisted artefacts (no experiments are run).
    """
    import sys

    # Make the package importable when invoked as a bare script.
    pkg_parent = Path(__file__).resolve().parent.parent.parent
    if str(pkg_parent) not in sys.path:
        sys.path.insert(0, str(pkg_parent))

    from experiments.core.logging_setup import setup_logging
    from experiments.configs.settings import SETTINGS
    from experiments.reports.aggregate_report import build_aggregate_report

    setup_logging(level=SETTINGS.log_level)

    report_path = rebuild_latest_report()
    agg_path = build_aggregate_report()

    if report_path is None and agg_path is None:
        logger.error("Nothing to build — run experiments first.")
        return 1
    if report_path:
        logger.info("Per-run report: %s", report_path)
    if agg_path:
        logger.info("Aggregated report: %s", agg_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
