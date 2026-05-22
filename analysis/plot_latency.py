"""
analysis/plot_latency.py
=========================
Generate publication-quality figures for the thesis from stored benchmark
and evaluation results.

Produces:
    figures/retrieval_latency_boxplot.png   — latency distributions per benchmark
    figures/generation_latency_boxplot.png  — generation latency distributions
    figures/corpus_scaling.png              — retrieval latency vs corpus size
    figures/concurrency_scaling.png         — latency and throughput vs concurrency
    figures/topk_quality.png                — retrieval quality metrics vs top-k
    figures/topk_latency.png                — retrieval latency vs top-k

Requires: matplotlib (pip install matplotlib)
Optional: numpy (for jitter in scatter plots)

Usage:
    cd C:\\rag-toolkit
    python analysis/plot_latency.py [OPTIONS]

Options:
    --benchmark-dir PATH  [results/benchmarks]
    --eval-dir PATH       [results/eval]
    --output-dir PATH     [analysis/figures]
    --dpi INT             Figure DPI [150]
    --format STR          Output format: png, pdf, svg [png]
    --style STR           Matplotlib style: seaborn-v0_8, ggplot, default [default]
    --no-show             Do not call plt.show() (useful in headless environments)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK_DIR = _ROOT / "results" / "benchmarks"
DEFAULT_EVAL_DIR      = _ROOT / "results" / "eval"
DEFAULT_OUTPUT_DIR    = _ROOT / "analysis" / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib import guard
# ─────────────────────────────────────────────────────────────────────────────

def _require_matplotlib():
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except ImportError:
        logger.error(
            "matplotlib is not installed. Install it with:\n"
            "    pip install matplotlib\n"
            "Then re-run this script."
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _stream_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                pass


def _load_benchmarks(benchmark_dir: Path) -> List[Dict[str, Any]]:
    """Load all benchmark records from all JSONL files."""
    records: List[Dict[str, Any]] = []
    for path in sorted(benchmark_dir.glob("*.jsonl")):
        for rec in _stream_jsonl(path):
            if rec.get("type") == "benchmark":
                records.append(rec)
    return records


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load a CSV file into a list of dicts with numeric coercion."""
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coerced: Dict[str, Any] = {}
            for k, v in row.items():
                try:
                    coerced[k] = float(v)
                except (ValueError, TypeError):
                    coerced[k] = v
            rows.append(coerced)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2196F3", "#4CAF50", "#F44336", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548", "#607D8B",
]


# ─────────────────────────────────────────────────────────────────────────────
# Plot: latency box plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_boxplot(
    data: Dict[str, List[float]],
    title: str,
    ylabel: str,
    output_path: Path,
    plt,
    dpi: int,
    show: bool,
) -> None:
    """Create a box plot of latency distributions, one box per key in data."""
    if not data:
        logger.info("Skipping boxplot — no data: %s", title)
        return

    labels = list(data.keys())
    values = [data[k] for k in labels]

    # Skip empty series
    labels = [l for l, v in zip(labels, values) if v]
    values = [v for v in values if v]
    if not labels:
        logger.info("Skipping boxplot — all series are empty: %s", title)
        return

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 2.5), 6))

    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )

    for patch, colour in zip(bp["boxes"], _PALETTE):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Configuration", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="x", rotation=15 if max(len(l) for l in labels) > 15 else 0)

    # Annotate medians
    for i, (vals, med_line) in enumerate(zip(values, bp["medians"])):
        med = sorted(vals)[len(vals) // 2]
        ax.text(
            i + 1, med * 1.02, f"{med:.0f}",
            ha="center", va="bottom", fontsize=8, color="black",
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: scaling line charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_scaling_line(
    rows: List[Dict[str, Any]],
    x_col: str,
    y_cols: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    plt,
    dpi: int,
    show: bool,
    y_secondary: Optional[str] = None,
    y_secondary_label: str = "",
) -> None:
    """Create a line chart for scaling experiments."""
    if not rows:
        logger.info("Skipping line chart — no data: %s", title)
        return

    x = [r[x_col] for r in rows if x_col in r]
    if not x:
        logger.info("Skipping line chart — x column '%s' missing: %s", x_col, title)
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    for col, colour in zip(y_cols, _PALETTE):
        y = [r.get(col) for r in rows]
        if all(v is None for v in y):
            continue
        ax1.plot(
            x, y,
            marker="o", linewidth=2, label=col,
            color=colour, markersize=6,
        )

    ax1.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.grid(linestyle="--", alpha=0.6)

    if y_secondary and any(y_secondary in r for r in rows):
        ax2 = ax1.twinx()
        y2 = [r.get(y_secondary) for r in rows]
        ax2.plot(
            x, y2,
            marker="s", linewidth=2, linestyle="--",
            label=y_secondary_label or y_secondary,
            color=_PALETTE[len(y_cols) % len(_PALETTE)],
            markersize=6,
        )
        ax2.set_ylabel(y_secondary_label or y_secondary, fontsize=11)
        ax2.legend(loc="upper right", fontsize=9)

    ax1.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Plot: concurrency (dual axis)
# ─────────────────────────────────────────────────────────────────────────────

def plot_concurrency(
    benchmarks: List[Dict[str, Any]],
    output_path: Path,
    plt,
    dpi: int,
    show: bool,
) -> None:
    """Plot latency and throughput vs concurrency level."""
    conc_records = [
        b for b in benchmarks
        if b.get("concurrency") is not None
        and b.get("stats", {}).get("n", 0) > 0
    ]
    if not conc_records:
        logger.info("Skipping concurrency plot — no concurrency benchmark records found.")
        return

    # Sort by concurrency level
    conc_records.sort(key=lambda b: b.get("concurrency", 0))

    x    = [b["concurrency"] for b in conc_records]
    mean = [b["stats"]["mean"]   for b in conc_records]
    p95  = [b["stats"]["p95"]    for b in conc_records]
    tput = [
        b["stats"]["n"] / max(b.get("duration_s", 1), 1e-6)
        for b in conc_records
    ]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    lns1 = ax1.plot(x, mean, marker="o", linewidth=2, color=_PALETTE[0], label="Mean latency (ms)")
    lns2 = ax1.plot(x, p95,  marker="s", linewidth=2, color=_PALETTE[2],
                    linestyle="--", label="p95 latency (ms)")
    ax1.set_xlabel("Concurrent workers", fontsize=11)
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.grid(linestyle="--", alpha=0.6)

    ax2 = ax1.twinx()
    lns3 = ax2.plot(x, tput, marker="^", linewidth=2, color=_PALETTE[1],
                    linestyle=":", label="Throughput (req/s)")
    ax2.set_ylabel("Throughput (req/s)", fontsize=11)

    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left", fontsize=9)

    ax1.set_title("Retrieval Latency and Throughput vs Concurrency", fontsize=13,
                  fontweight="bold", pad=12)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    logger.info("Saved: %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate thesis figures from benchmark results")
    p.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    p.add_argument("--eval-dir",      type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--output-dir",    type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--dpi",           type=int,  default=150)
    p.add_argument("--format",        type=str,  default="png",
                   choices=["png", "pdf", "svg"])
    p.add_argument("--style",         type=str,  default="default")
    p.add_argument("--no-show",       action="store_true",
                   help="Suppress plt.show() — use in headless / CI environments")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    matplotlib, plt = _require_matplotlib()

    # Apply style if available
    try:
        if args.style != "default":
            plt.style.use(args.style)
    except OSError:
        logger.warning("Style '%s' not available — using default.", args.style)

    show = not args.no_show
    ext  = args.format
    out  = args.output_dir
    dpi  = args.dpi

    def _out(name: str) -> Path:
        return out / f"{name}.{ext}"

    logger.info("Loading benchmark records from: %s", args.benchmark_dir)
    benchmarks = _load_benchmarks(args.benchmark_dir)
    logger.info("Found %d benchmark records", len(benchmarks))

    # ── 1. Retrieval latency boxplot ──────────────────────────────────────────
    retrieval_benches = [
        b for b in benchmarks
        if b.get("config", {}).get("benchmark_type") == "retrieval"
        or "retrieval" in b.get("benchmark_name", "")
    ]
    ret_data: Dict[str, List[float]] = {}
    for b in retrieval_benches:
        name = b.get("benchmark_name", "unknown")
        raw  = b.get("raw_values", [])
        if raw:
            # Shorten name for label
            short = name[:30]
            ret_data[short] = raw
    plot_latency_boxplot(
        ret_data,
        title="Retrieval Latency Distribution by Configuration",
        ylabel="Latency (ms)",
        output_path=_out("retrieval_latency_boxplot"),
        plt=plt, dpi=dpi, show=show,
    )

    # ── 2. Generation latency boxplot ─────────────────────────────────────────
    gen_benches = [
        b for b in benchmarks
        if b.get("config", {}).get("benchmark_type") == "generation"
        or "generation" in b.get("benchmark_name", "")
    ]
    gen_data: Dict[str, List[float]] = {}
    for b in gen_benches:
        name = b.get("benchmark_name", "unknown")
        raw  = b.get("raw_values", [])
        if raw:
            short = name[:30]
            gen_data[short] = raw
    plot_latency_boxplot(
        gen_data,
        title="Generation Latency Distribution (Mistral 7B via Ollama)",
        ylabel="Latency (ms)",
        output_path=_out("generation_latency_boxplot"),
        plt=plt, dpi=dpi, show=show,
    )

    # ── 3. Corpus scaling ─────────────────────────────────────────────────────
    scaling_csv = args.benchmark_dir / "corpus_scaling_summary.csv"
    scaling_rows = _load_csv(scaling_csv)
    if not scaling_rows:
        # Try to build summary rows from benchmark records
        corpus_benches = [
            b for b in benchmarks
            if b.get("corpus_size") is not None
            and "scaling" in b.get("benchmark_name", "").lower()
        ]
        if corpus_benches:
            corpus_benches.sort(key=lambda b: b.get("corpus_size", 0))
            scaling_rows = [
                {
                    "corpus_size": b["corpus_size"],
                    "mean_ms":     b["stats"]["mean"],
                    "median_ms":   b["stats"]["median"],
                    "p95_ms":      b["stats"]["p95"],
                    "std_dev_ms":  b["stats"]["std_dev"],
                }
                for b in corpus_benches
            ]

    plot_scaling_line(
        scaling_rows,
        x_col="corpus_size",
        y_cols=["mean_ms", "median_ms", "p95_ms"],
        title="Retrieval Latency vs Corpus Size",
        xlabel="Corpus Size (documents)",
        ylabel="Latency (ms)",
        output_path=_out("corpus_scaling"),
        plt=plt, dpi=dpi, show=show,
    )

    # ── 4. Concurrency scaling ────────────────────────────────────────────────
    plot_concurrency(
        benchmarks,
        output_path=_out("concurrency_scaling"),
        plt=plt, dpi=dpi, show=show,
    )

    # ── 5. Top-k quality ──────────────────────────────────────────────────────
    topk_csv = args.eval_dir / "topk_ablation_summary.csv"
    topk_rows = _load_csv(topk_csv)
    if topk_rows:
        plot_scaling_line(
            topk_rows,
            x_col="top_k",
            y_cols=["context_precision", "context_recall", "mrr", "ndcg"],
            title="Retrieval Quality Metrics vs Top-k",
            xlabel="Top-k",
            ylabel="Metric Score",
            output_path=_out("topk_quality"),
            plt=plt, dpi=dpi, show=show,
        )
        plot_scaling_line(
            topk_rows,
            x_col="top_k",
            y_cols=["mean_latency_ms"],
            title="Retrieval Latency vs Top-k",
            xlabel="Top-k",
            ylabel="Mean Latency (ms)",
            output_path=_out("topk_latency"),
            plt=plt, dpi=dpi, show=show,
        )
    else:
        logger.info(
            "Top-k ablation CSV not found (%s). "
            "Run experiments/09_topk_ablation/run_topk_ablation.py first.",
            topk_csv,
        )

    # ── 6. End-to-end benchmark ───────────────────────────────────────────────
    e2e_benches = [
        b for b in benchmarks
        if b.get("config", {}).get("benchmark_type") == "end_to_end"
        or "e2e" in b.get("benchmark_name", "").lower()
        or "end_to_end" in b.get("benchmark_name", "").lower()
    ]
    e2e_data: Dict[str, List[float]] = {}
    for b in e2e_benches:
        name = b.get("benchmark_name", "unknown")
        raw  = b.get("raw_values", [])
        if raw:
            e2e_data[name[:30]] = raw
    plot_latency_boxplot(
        e2e_data,
        title="End-to-End RAG Pipeline Latency Distribution",
        ylabel="Latency (ms)",
        output_path=_out("e2e_latency_boxplot"),
        plt=plt, dpi=dpi, show=show,
    )

    print(f"\nAll figures saved to: {out}")
    generated = list(out.glob(f"*.{ext}"))
    if generated:
        for p in sorted(generated):
            print(f"  {p.name}")
    else:
        print("  (no figures generated — run experiment scripts first to produce data)")

    logger.info("Plotting complete.")


if __name__ == "__main__":
    main()
