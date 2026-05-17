"""
analysis/analyse_results.py
============================
Generate thesis-ready comparison tables from stored evaluation results.

Reads EvaluationRunResult JSONL files from results/eval/ and BenchmarkResult
JSONL files from results/benchmarks/ and prints formatted ASCII tables.

Usage:
    cd C:\\rag-toolkit
    python analysis/analyse_results.py [OPTIONS]

Options:
    --eval-dir PATH       Directory containing evaluation JSONL files [results/eval]
    --benchmark-dir PATH  Directory containing benchmark JSONL files [results/benchmarks]
    --output-dir PATH     Directory for generated report files [analysis/reports]
    --list-runs           List all available run IDs without generating tables

The script auto-discovers all run_summary records in the eval directory and
all benchmark records in the benchmark directory, then prints:
    1. Retrieval metric comparison table
    2. Generation metric comparison table
    3. End-to-end metric comparison table
    4. Performance benchmark table
    5. Top-k ablation table (if topk_ablation_summary.csv exists)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_EVAL_DIR      = _ROOT / "results" / "eval"
DEFAULT_BENCHMARK_DIR = _ROOT / "results" / "benchmarks"
DEFAULT_OUTPUT_DIR    = _ROOT / "analysis" / "reports"


# ─────────────────────────────────────────────────────────────────────────────
# JSONL loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stream_jsonl(path: Path):
    """Yield all valid JSON records from a JSONL file."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.debug("Skipping malformed line %d in %s: %s", line_no, path, exc)


def _load_run_summaries(eval_dir: Path) -> List[Dict[str, Any]]:
    """Load all run_summary records from all JSONL files in eval_dir."""
    summaries: List[Dict[str, Any]] = []
    for jsonl_path in sorted(eval_dir.glob("*.jsonl")):
        for record in _stream_jsonl(jsonl_path):
            if record.get("type") == "run_summary":
                record["_source_file"] = jsonl_path.name
                summaries.append(record)
    return summaries


def _load_benchmark_records(benchmark_dir: Path) -> List[Dict[str, Any]]:
    """Load all benchmark records from all JSONL files in benchmark_dir."""
    records: List[Dict[str, Any]] = []
    for jsonl_path in sorted(benchmark_dir.glob("*.jsonl")):
        for record in _stream_jsonl(jsonl_path):
            if record.get("type") == "benchmark":
                record["_source_file"] = jsonl_path.name
                records.append(record)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Metric extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_value(summary: Dict[str, Any], name: str) -> Optional[float]:
    """Extract a scalar metric value from a run_summary record."""
    metrics = summary.get("metrics", {})
    # Exact match
    if name in metrics:
        val = metrics[name]
        return val.get("value") if isinstance(val, dict) else float(val)
    # Prefix match (e.g. 'ndcg' matches 'ndcg@10')
    for key, val in metrics.items():
        if key.startswith(name):
            return val.get("value") if isinstance(val, dict) else float(val)
    return None


def _fmt(val: Optional[float], decimals: int = 4) -> str:
    """Format a metric value or return '—' when unavailable."""
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


# ─────────────────────────────────────────────────────────────────────────────
# Table printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _col_width(header: str, rows: List[str]) -> int:
    return max(len(header), *(len(r) for r in rows))


def _print_table(
    title: str,
    columns: List[str],
    rows: List[List[str]],
) -> None:
    """Print a formatted ASCII table."""
    if not rows:
        print(f"\n[{title}]\n  No data available.\n")
        return

    widths = [max(len(col), max(len(r[i]) for r in rows)) for i, col in enumerate(columns)]
    sep = "─" * (sum(widths) + 3 * len(widths) + 1)

    print(f"\n{'═' * len(sep)}")
    print(f"  {title}")
    print(f"{'═' * len(sep)}")
    header_row = "  " + "   ".join(col.ljust(widths[i]) for i, col in enumerate(columns))
    print(header_row)
    print("  " + sep)
    for row in rows:
        print("  " + "   ".join(str(row[i]).ljust(widths[i]) for i in range(len(columns))))
    print()


def _write_table_csv(
    path: Path,
    columns: List[str],
    rows: List[List[str]],
) -> None:
    """Write a table to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    logger.info("Written: %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# Table generators
# ─────────────────────────────────────────────────────────────────────────────

def table_retrieval_comparison(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Print and save retrieval metric comparison table."""
    retrieval_runs = [s for s in summaries if s.get("config", {}).get("mode") == "retrieval"]
    if not retrieval_runs:
        # Fall back to end-to-end summaries that include retrieval metrics
        retrieval_runs = [
            s for s in summaries
            if s.get("config", {}).get("mode") in {"end_to_end", "retrieval"}
        ]

    columns = ["Run ID", "top_k", "Precision", "Recall", "MRR", "nDCG", "Latency ms"]
    rows: List[List[str]] = []

    for s in retrieval_runs:
        cfg = s.get("config", {})
        rows.append([
            s.get("run_id", "—")[:40],
            str(cfg.get("top_k", "—")),
            _fmt(_metric_value(s, "context_precision")),
            _fmt(_metric_value(s, "context_recall")),
            _fmt(_metric_value(s, "mrr")),
            _fmt(_metric_value(s, "ndcg")),
            _fmt(_metric_value(s, "latency"), 1),
        ])

    _print_table("TABLE 1: Retrieval Quality — Configuration Comparison", columns, rows)
    _write_table_csv(output_dir / "table1_retrieval_comparison.csv", columns, rows)


def table_generation_comparison(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Print and save generation metric comparison table."""
    gen_runs = [
        s for s in summaries
        if s.get("config", {}).get("mode") in {"generation", "end_to_end"}
        and _metric_value(s, "exact_match") is not None
    ]

    columns = ["Run ID", "mode", "Exact Match", "Token F1", "Precision", "Recall", "Latency ms"]
    rows: List[List[str]] = []

    for s in gen_runs:
        cfg = s.get("config", {})
        tf1_meta = s.get("metrics", {}).get("token_f1", {})
        if isinstance(tf1_meta, dict):
            meta = tf1_meta.get("metadata", {})
        else:
            meta = {}
        rows.append([
            s.get("run_id", "—")[:40],
            cfg.get("mode", "—"),
            _fmt(_metric_value(s, "exact_match")),
            _fmt(_metric_value(s, "token_f1")),
            _fmt(meta.get("mean_precision")),
            _fmt(meta.get("mean_recall")),
            _fmt(_metric_value(s, "latency"), 1),
        ])

    _print_table("TABLE 2: Generation Quality — Configuration Comparison", columns, rows)
    _write_table_csv(output_dir / "table2_generation_comparison.csv", columns, rows)


def table_e2e_comparison(
    summaries: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Print and save end-to-end metric comparison table."""
    e2e_runs = [
        s for s in summaries
        if s.get("config", {}).get("mode") == "end_to_end"
    ]

    columns = [
        "Run ID", "top_k", "Precision", "Recall",
        "MRR", "nDCG", "EM", "Token F1", "E2E ms",
    ]
    rows: List[List[str]] = []

    for s in e2e_runs:
        cfg = s.get("config", {})
        lat_meta = s.get("metrics", {}).get("latency", {})
        if isinstance(lat_meta, dict):
            e2e_ms = lat_meta.get("metadata", {}).get("mean_end_to_end_ms")
        else:
            e2e_ms = None

        rows.append([
            s.get("run_id", "—")[:36],
            str(cfg.get("top_k", "—")),
            _fmt(_metric_value(s, "context_precision")),
            _fmt(_metric_value(s, "context_recall")),
            _fmt(_metric_value(s, "mrr")),
            _fmt(_metric_value(s, "ndcg")),
            _fmt(_metric_value(s, "exact_match")),
            _fmt(_metric_value(s, "token_f1")),
            _fmt(e2e_ms, 1),
        ])

    _print_table("TABLE 3: End-to-End RAG Evaluation — Full Metric Overview", columns, rows)
    _write_table_csv(output_dir / "table3_e2e_comparison.csv", columns, rows)


def table_benchmark_comparison(
    benchmark_records: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """Print and save benchmark statistics table."""
    columns = [
        "Benchmark", "corpus_size", "top_k", "concurrency",
        "Mean ms", "Median ms", "p95 ms", "StdDev ms", "n",
    ]
    rows: List[List[str]] = []

    for rec in benchmark_records:
        stats = rec.get("stats", {})
        rows.append([
            rec.get("benchmark_name", "—")[:36],
            str(rec.get("corpus_size") or "—"),
            str(rec.get("top_k") or "—"),
            str(rec.get("concurrency") or "—"),
            _fmt(stats.get("mean"), 1),
            _fmt(stats.get("median"), 1),
            _fmt(stats.get("p95"), 1),
            _fmt(stats.get("std_dev"), 1),
            str(stats.get("n", "—")),
        ])

    _print_table("TABLE 4: Performance Benchmarks — Latency Statistics", columns, rows)
    _write_table_csv(output_dir / "table4_benchmarks.csv", columns, rows)


def table_topk_ablation(
    eval_dir: Path,
    output_dir: Path,
) -> None:
    """Print top-k ablation table from summary CSV if it exists."""
    csv_path = eval_dir / "topk_ablation_summary.csv"
    if not csv_path.exists():
        logger.info("Top-k ablation CSV not found (%s) — skipping.", csv_path)
        return

    rows_raw: List[List[str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            rows_raw.append([row.get(fn, "—") for fn in fieldnames])

    columns = list(fieldnames)
    _print_table("TABLE 5: Top-k Ablation Study", columns, rows_raw)
    _write_table_csv(output_dir / "table5_topk_ablation.csv", columns, rows_raw)


# ─────────────────────────────────────────────────────────────────────────────
# Listing helper
# ─────────────────────────────────────────────────────────────────────────────

def list_available_runs(summaries: List[Dict[str, Any]], benchmarks: List[Dict[str, Any]]) -> None:
    """Print all available run IDs."""
    print("\n── Evaluation Runs ─────────────────────────────────────────────────────────────")
    if not summaries:
        print("  (none found)")
    else:
        for s in summaries:
            cfg = s.get("config", {})
            print(
                f"  {s.get('run_id', '?'):<48}  mode={cfg.get('mode','?'):<12}  "
                f"n={s.get('n_examples','?')}  ts={s.get('timestamp_iso','?')[:19]}"
            )

    print("\n── Benchmark Runs ──────────────────────────────────────────────────────────────")
    if not benchmarks:
        print("  (none found)")
    else:
        for b in benchmarks:
            stats = b.get("stats", {})
            print(
                f"  {b.get('benchmark_name','?'):<48}  "
                f"n={stats.get('n','?')}  mean={stats.get('mean','?')} ms  "
                f"ts={b.get('timestamp_iso','?')[:19]}"
            )
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis tables from evaluation results")
    p.add_argument("--eval-dir",       type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--benchmark-dir",  type=Path, default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--output-dir",     type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--list-runs",      action="store_true",
                   help="List all available runs without generating tables")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading evaluation summaries from: %s", args.eval_dir)
    summaries = _load_run_summaries(args.eval_dir)
    logger.info("Found %d run_summary records", len(summaries))

    logger.info("Loading benchmark records from: %s", args.benchmark_dir)
    benchmarks = _load_benchmark_records(args.benchmark_dir)
    logger.info("Found %d benchmark records", len(benchmarks))

    if args.list_runs:
        list_available_runs(summaries, benchmarks)
        return

    if not summaries and not benchmarks:
        print(
            "\nNo evaluation or benchmark results found.\n"
            "Run the experiment scripts first, then re-run this analysis.\n"
            f"  Eval dir:      {args.eval_dir}\n"
            f"  Benchmark dir: {args.benchmark_dir}\n"
        )
        return

    # ── Generate tables ───────────────────────────────────────────────────────
    table_retrieval_comparison(summaries, args.output_dir)
    table_generation_comparison(summaries, args.output_dir)
    table_e2e_comparison(summaries, args.output_dir)
    table_benchmark_comparison(benchmarks, args.output_dir)
    table_topk_ablation(args.eval_dir, args.output_dir)

    print(f"All tables saved to: {args.output_dir}")
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
