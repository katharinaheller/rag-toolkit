"""One-command entry point for the experiments framework.

Usage examples::

    python -m experiments.run_all_experiments
    python -m experiments.run_all_experiments --only-suites s01,s02,s05
    python -m experiments.run_all_experiments --corpora n10,n100 --n-queries 30

The runner is intentionally linear so you can read it top to bottom:

1. parse arguments / read environment;
2. set up logging and mint a run_id;
3. load each corpus's chunks (cached via the ingestion pipeline);
4. generate synthetic queries per corpus (deterministic seed);
5. build the generator (optional, gracefully skipped if Ollama is down);
6. instantiate and execute every selected suite;
7. emit a dashboard plot, the Markdown report and a run manifest.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from experiments.adapters.generation_adapter import build_generator
from experiments.configs.default_matrix import CORPORA
from experiments.configs.settings import SETTINGS, output_subdir
from experiments.core.corpus_loader import load_corpus_chunks
from experiments.core.logging_setup import get_logger, new_run_id, setup_logging
from experiments.core.query_generator import generate_queries
from experiments.reports import build_report
from experiments.storage import write_manifest
from experiments.suites import ALL_SUITES, ExperimentContext, Suite
from experiments.visualisation import plot_dashboard, setup_matplotlib


logger = get_logger(__name__)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full RAG experiments framework."
    )
    parser.add_argument(
        "--only-suites", default="",
        help=("Comma-separated suite keys or short ids to run "
              "(e.g. 's01,s05' or 's01_retriever_corpus_scaling'). "
              "Empty = all suites."),
    )
    parser.add_argument(
        "--corpora", default="",
        help=("Comma-separated corpus names. Empty = "
              f"{','.join(CORPORA)}."),
    )
    parser.add_argument(
        "--n-queries", type=int, default=SETTINGS.n_queries_per_corpus,
        help="Synthetic queries per corpus (default: %(default)s).",
    )
    parser.add_argument(
        "--seed", type=int, default=SETTINGS.seed,
        help="Random seed for deterministic query generation.",
    )
    parser.add_argument(
        "--enable-generation", action="store_true",
        help="Force-enable generation suites (overrides env var).",
    )
    parser.add_argument(
        "--disable-generation", action="store_true",
        help="Force-disable generation suites (overrides env var).",
    )
    parser.add_argument(
        "--log-level", default=SETTINGS.log_level,
        help="Logging level (default: %(default)s).",
    )
    return parser.parse_args(argv)


def _filter_suites(only: str) -> List[Type[Suite]]:
    if not only.strip():
        return list(ALL_SUITES)
    wanted = [s.strip().lower() for s in only.split(",") if s.strip()]
    selected: List[Type[Suite]] = []
    for suite_cls in ALL_SUITES:
        key = suite_cls.key.lower()
        short = key.split("_", 1)[0]  # e.g. "s05"
        if any(w == key or w == short or w in key for w in wanted):
            selected.append(suite_cls)
    if not selected:
        logger.warning("No suites matched filter %r; running nothing.", only)
    return selected


def _resolve_corpora(args_corpora: str) -> List[str]:
    if args_corpora.strip():
        return [c.strip() for c in args_corpora.split(",") if c.strip()]
    return list(CORPORA)


def _load_corpora(corpora: List[str]) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for name in corpora:
        try:
            chunks = load_corpus_chunks(name)
            out[name] = chunks
            logger.info("Loaded corpus %s: %d chunks", name, len(chunks))
        except Exception as exc:
            logger.warning("Failed to load corpus %s: %s", name, exc)
    return out


def _generate_all_queries(
    corpora_chunks: Dict[str, List[Dict]],
    n_per_corpus: int,
    seed: int,
) -> Dict[str, list]:
    out: Dict[str, list] = {}
    for name, chunks in corpora_chunks.items():
        try:
            queries = generate_queries(chunks, name, n_total=n_per_corpus, seed=seed)
            out[name] = queries
            logger.info("Generated %d queries for %s", len(queries), name)
        except Exception as exc:
            logger.warning("Query generation failed for %s: %s", name, exc)
            out[name] = []
    return out


def _settings_dict(args: argparse.Namespace) -> Dict[str, Any]:
    sd = dataclasses.asdict(SETTINGS)
    # Path objects → str for JSON serialisation.
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in sd.items()}


def _build_dashboard_summary(summaries: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
    """Project per-suite rows into the four panels the dashboard expects."""
    by_key = {s.get("key", ""): s for s in summaries}
    out: Dict[str, List[Dict]] = {
        "scaling": [], "topk": [], "pareto": [], "stability": [],
    }

    s1 = by_key.get("s01_retriever_corpus_scaling", {})
    for row in s1.get("rows", []):
        # ``recall@10`` is present from the metric aggregator.
        if "recall@10" not in row:
            continue
        out["scaling"].append({
            "retriever": row["retriever"],
            "corpus_size": row.get("corpus_size", 0),
            "recall@10": row["recall@10"],
        })

    s2 = by_key.get("s02_topk_sensitivity", {})
    for row in s2.get("rows", []):
        out["topk"].append({
            "retriever": row["retriever"],
            "top_k": row["top_k"],
            "recall": row["recall"],
        })

    s5 = by_key.get("s05_latency_quality_pareto", {})
    largest_corpus = None
    if s5.get("rows"):
        largest_corpus = max(
            {(r.get("corpus"), r.get("corpus_size", 0)) for r in s5["rows"]},
            key=lambda t: t[1],
        )[0]
    for row in s5.get("rows", []):
        if largest_corpus is not None and row.get("corpus") != largest_corpus:
            continue
        out["pareto"].append({
            "retriever": row["retriever"],
            "latency_ms": row["mean_latency_ms"],
            "recall": row.get("recall@10", 0.0),
            "label": row["retriever"],
        })

    s9 = by_key.get("s09_stability", {})
    for row in s9.get("rows", []):
        out["stability"].append({
            "retriever": row["retriever"],
            "stability": row.get("rank_stability", 0.0),
        })
    return out


def run(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Apply CLI-only overrides on top of env-driven SETTINGS.
    enable_generation = SETTINGS.enable_generation
    if args.enable_generation:
        enable_generation = True
    if args.disable_generation:
        enable_generation = False

    run_id = new_run_id("run")
    log_dir = output_subdir("logs", run_id)
    logfile = log_dir / "run.log"
    setup_logging(level=args.log_level, logfile=logfile)
    setup_matplotlib()

    logger.info("=" * 70)
    logger.info("Starting run %s", run_id)
    logger.info("=" * 70)

    corpora = _resolve_corpora(args.corpora)
    logger.info("Corpora: %s", corpora)
    corpora_chunks = _load_corpora(corpora)
    if not corpora_chunks:
        logger.error("No corpora loaded — aborting.")
        return 2

    queries = _generate_all_queries(
        corpora_chunks, n_per_corpus=args.n_queries, seed=args.seed,
    )

    # Build generator. Honour the runtime override by syncing SETTINGS so the
    # adapter sees the effective value (CLI flags take precedence over env).
    import os as _os
    from experiments.adapters.generation_adapter import BuiltGenerator
    if args.enable_generation:
        _os.environ["EXPERIMENTS_ENABLE_GENERATION"] = "1"
        object.__setattr__(SETTINGS, "enable_generation", True)
    if args.disable_generation:
        _os.environ["EXPERIMENTS_ENABLE_GENERATION"] = "0"
        object.__setattr__(SETTINGS, "enable_generation", False)

    if enable_generation:
        generator = build_generator()
    else:
        generator = BuiltGenerator(
            strategy=None,
            available=False,
            reason="generation disabled via configuration (env/CLI)",
            model_name=SETTINGS.ollama_model,
        )

    if generator.available:
        logger.info("Generator available: model=%s", generator.model_name)
    else:
        logger.info("Generator disabled or unreachable: %s", generator.reason)

    ctx = ExperimentContext(
        run_id=run_id,
        corpora_chunks=corpora_chunks,
        queries=queries,
        generator=generator,
        metadata={
            "corpora": corpora,
            "n_queries_per_corpus": args.n_queries,
            "seed": args.seed,
            "generation_enabled": generator.available,
        },
    )

    suites_to_run = _filter_suites(args.only_suites)
    logger.info("Running %d suites: %s",
                len(suites_to_run), [s.key for s in suites_to_run])

    summaries: List[Dict[str, Any]] = []
    for suite_cls in suites_to_run:
        suite = suite_cls(ctx)
        summary = suite.execute()
        summaries.append(summary)

    # Dashboard.
    dashboard_panels = _build_dashboard_summary(summaries)
    report_dir = output_subdir("reports", run_id)
    dashboard_path = report_dir / "dashboard.png"
    try:
        plot_dashboard(dashboard_panels, dashboard_path)
    except Exception as exc:
        logger.warning("Dashboard rendering failed: %s", exc)
        dashboard_path = None

    # Final report.
    settings_dict = _settings_dict(args)
    settings_dict.update({
        "n_queries_per_corpus": args.n_queries,
        "seed": args.seed,
        "enable_generation": generator.available,
    })
    report_path = build_report(
        run_id=run_id,
        summaries=summaries,
        settings_dict=settings_dict,
        report_dir=report_dir,
        dashboard_figure=dashboard_path if dashboard_path and dashboard_path.exists() else None,
    )

    # Run manifest.
    manifest_path = output_subdir("logs", run_id) / "manifest.json"
    write_manifest(
        manifest_path,
        run_id=run_id,
        settings_dict=settings_dict,
        suites_planned=[s.key for s in suites_to_run],
        extra={
            "suite_results": [
                {
                    "key": s.get("key"),
                    "status": s.get("status"),
                    "duration_s": s.get("duration_s"),
                    "figures": s.get("figures", []),
                    "tables": s.get("tables", []),
                    "n_findings": len(s.get("findings", [])),
                }
                for s in summaries
            ],
            "report": str(report_path),
            "dashboard": str(dashboard_path) if dashboard_path else None,
        },
    )

    logger.info("=" * 70)
    logger.info("Run %s complete", run_id)
    logger.info("Report:    %s", report_path)
    logger.info("Manifest:  %s", manifest_path)
    if dashboard_path:
        logger.info("Dashboard: %s", dashboard_path)
    logger.info("=" * 70)

    # Exit code: 1 if every suite failed, else 0.
    if summaries and all(s.get("status") == "failed" for s in summaries):
        return 1
    return 0


def main() -> None:  # entry point for `python -m experiments.run_all_experiments`
    sys.exit(run())


if __name__ == "__main__":
    main()
