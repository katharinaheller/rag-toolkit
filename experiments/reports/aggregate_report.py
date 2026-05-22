"""Cross-run aggregated report.

Walks every run under ``outputs/aggregated/`` and produces a single
``AGGREGATE_REPORT.md`` plus an ``aggregate_benchmarks.csv`` that stacks the
benchmark CSVs from all runs. This is what lets you compare a CPU-only run to a
later GPU run, or track regression across runs.

Pure stdlib + the project's storage helpers; matplotlib is optional.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.storage import write_csv

logger = logging.getLogger(__name__)


_BENCH_SUITES = (
    "s16_gpu_embedding_benchmark",
    "s17_gpu_generation_benchmark",
    "s18_batchsize_scaling",
    "s19_cold_warm",
    "s20_resource_timeline",
    "s21_concurrency_scaling",
)


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    import csv
    if not path.exists():
        return []
    out: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.append(dict(row))
    except Exception as exc:
        logger.warning("Failed to read CSV %s: %s", path, exc)
    return out


def _discover_runs(output_root: Path) -> List[str]:
    agg = output_root / "aggregated"
    if not agg.exists():
        return []
    return sorted(p.name for p in agg.iterdir() if p.is_dir())


def _load_summary(output_root: Path, run_id: str, suite_key: str) -> Optional[Dict[str, Any]]:
    path = output_root / "aggregated" / run_id / suite_key / "_suite_summary.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_aggregate_report(output_root: Optional[Path] = None) -> Optional[Path]:
    """Build the cross-run aggregated report. Returns its path or None."""
    from experiments.configs.settings import SETTINGS

    root = Path(output_root) if output_root else SETTINGS.output_root
    runs = _discover_runs(root)
    if not runs:
        logger.warning("No runs to aggregate under %s.", root)
        return None

    out_dir = root / "reports" / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Stack every benchmark CSV across all runs.
    stacked: List[Dict[str, Any]] = []
    benchmark_csv_names = {
        "s16_gpu_embedding_benchmark": "embedding_benchmark.csv",
        "s17_gpu_generation_benchmark": "generation_benchmark.csv",
        "s18_batchsize_scaling": "batchsize_scaling.csv",
        "s19_cold_warm": "cold_warm.csv",
        "s20_resource_timeline": "resource_timeline_summary.csv",
        "s21_concurrency_scaling": "concurrency_scaling.csv",
    }
    for run_id in runs:
        for suite_key, csv_name in benchmark_csv_names.items():
            csv_path = root / "aggregated" / run_id / suite_key / csv_name
            for row in _read_csv_rows(csv_path):
                stacked.append({"run_id": run_id, "suite": suite_key, **row})

    agg_csv = out_dir / "aggregate_benchmarks.csv"
    write_csv(agg_csv, stacked)

    # 2) Markdown summary.
    lines: List[str] = []
    lines.append("# RAG Experiments — Aggregated Report")
    lines.append("")
    lines.append(f"_Generated: {datetime.datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append(f"Runs aggregated: **{len(runs)}**")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    for run_id in runs:
        lines.append(f"- `{run_id}`")
    lines.append("")

    lines.append("## Benchmark Headlines per Run")
    lines.append("")
    lines.append("| Run | Suite | Status | Headline finding |")
    lines.append("|-----|-------|--------|------------------|")
    for run_id in runs:
        for suite_key in _BENCH_SUITES:
            summary = _load_summary(root, run_id, suite_key)
            if not summary:
                continue
            status = summary.get("status", "?")
            findings = summary.get("findings", [])
            headline = findings[0] if findings else "—"
            headline = headline.replace("|", "\\|").replace("\n", " ")
            short = suite_key.split("_", 1)[0]
            lines.append(f"| `{run_id}` | {short} | {status} | {headline} |")
    lines.append("")

    lines.append("## Aggregated Benchmark Table")
    lines.append("")
    lines.append(
        f"All benchmark rows from every run are stacked in "
        f"[`{agg_csv.name}`]({agg_csv.name}) ({len(stacked)} rows)."
    )
    lines.append("")

    chart_path = _try_plot_cross_run_speedup(root, runs, out_dir)
    if chart_path is not None:
        lines.append("## Cross-run GPU Embedding Speedup")
        lines.append("")
        lines.append(f"![cross run speedup]({chart_path.name})")
        lines.append("")

    report_path = out_dir / "AGGREGATE_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Aggregated report written: %s", report_path)
    return report_path


def _try_plot_cross_run_speedup(
    root: Path, runs: List[str], out_dir: Path,
) -> Optional[Path]:
    """If runs captured both CPU and GPU embedding rows, plot speedup per run."""
    try:
        from experiments.visualisation.style import save_fig, setup_matplotlib
        if not setup_matplotlib():
            return None
        import matplotlib.pyplot as plt
    except Exception:
        return None

    per_run: Dict[str, float] = {}
    for run_id in runs:
        csv_path = (root / "aggregated" / run_id
                    / "s16_gpu_embedding_benchmark" / "embedding_benchmark.csv")
        rows = _read_csv_rows(csv_path)
        cpu_lat: List[float] = []
        gpu_lat: List[float] = []
        for r in rows:
            raw = r.get("stats.mean", "")
            try:
                mean = float(raw)
            except (TypeError, ValueError):
                continue
            device = r.get("device", "")
            if device == "cpu":
                cpu_lat.append(mean)
            elif device.startswith("cuda"):
                gpu_lat.append(mean)
        if cpu_lat and gpu_lat:
            cpu_mean = sum(cpu_lat) / len(cpu_lat)
            gpu_mean = sum(gpu_lat) / len(gpu_lat)
            if gpu_mean > 0:
                per_run[run_id] = round(cpu_mean / gpu_mean, 3)

    if not per_run:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(per_run) * 1.1), 5))
    labels = list(per_run.keys())
    values = [per_run[k] for k in labels]
    colours = ["#2ca02c" if v >= 1.0 else "#d62728" for v in values]
    ax.bar(range(len(labels)), values, color=colours, alpha=0.85)
    ax.axhline(1.0, color="#444", linestyle="--", linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Mean GPU speedup over CPU")
    ax.set_title("Cross-run GPU embedding speedup")
    out_path = out_dir / "cross_run_speedup.png"
    save_fig(fig, out_path)
    return out_path
