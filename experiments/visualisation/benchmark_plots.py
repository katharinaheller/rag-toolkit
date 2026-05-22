"""Benchmark visualisations.

Throughput-vs-batch-size, throughput-vs-latency, latency distributions
(histogram + percentile markers), CPU-vs-GPU grouped bars, concurrency
scaling, and cold-vs-warm comparisons. Same return-path conventions as the
rest of the visualisation layer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from experiments.visualisation.resource_plots import colour_for_device
from experiments.visualisation.style import save_fig, setup_matplotlib

logger = logging.getLogger(__name__)


def _plt():
    if not setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    return plt


# ── Throughput vs batch size ────────────────────────────────────────────────
def plot_throughput_vs_batchsize(
    rows: List[Dict],
    out_path: Path,
    title: str = "Throughput vs batch size",
    y_field: str = "items_per_second",
) -> Optional[Path]:
    """Line plot of throughput against batch size, one line per device."""
    plt = _plt()
    if plt is None or not rows:
        return None
    by_device: Dict[str, List[Tuple[int, float]]] = {}
    for r in rows:
        y = r.get(y_field)
        if y is None:
            continue
        bs = int(r.get("batch_size", 0))
        by_device.setdefault(str(r.get("device", "?")), []).append((bs, float(y)))
    if not by_device:
        return None
    fig, ax = plt.subplots()
    for device, pts in sorted(by_device.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", label=device, color=colour_for_device(device), linewidth=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Documents per second")
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, out_path)
    return out_path


# ── Throughput vs latency scatter ───────────────────────────────────────────
def plot_throughput_vs_latency(
    rows: List[Dict],
    out_path: Path,
    title: str = "Throughput vs latency",
) -> Optional[Path]:
    """Scatter of throughput (y) against mean latency (x), coloured by device."""
    plt = _plt()
    if plt is None or not rows:
        return None
    fig, ax = plt.subplots()
    by_device: Dict[str, List[Tuple[float, float, str]]] = {}
    for r in rows:
        lat = r.get("mean_latency_ms")
        tput = r.get("throughput_qps") or r.get("items_per_second")
        if lat is None or tput is None:
            continue
        device = str(r.get("device", "?"))
        by_device.setdefault(device, []).append(
            (float(lat), float(tput), str(r.get("label", "")))
        )
    if not by_device:
        return None
    for device, pts in sorted(by_device.items()):
        ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                   label=device, color=colour_for_device(device), s=50, alpha=0.8)
        for x, y, lab in pts:
            if lab:
                ax.annotate(lab, (x, y), xytext=(3, 3),
                            textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Throughput")
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, out_path)
    return out_path


# ── Latency distribution ────────────────────────────────────────────────────
def plot_latency_distribution(
    series_by_label: Dict[str, List[float]],
    out_path: Path,
    title: str = "Latency distribution",
    bins: int = 30,
) -> Optional[Path]:
    """Overlaid histograms with p50/p95/p99 markers per labelled series."""
    plt = _plt()
    if plt is None or not series_by_label:
        return None
    from experiments.core.benchmark_stats import percentile

    fig, ax = plt.subplots(figsize=(9, 5.5))
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    plotted = False
    for i, (label, values) in enumerate(sorted(series_by_label.items())):
        vals = [v for v in values if v is not None]
        if not vals:
            continue
        plotted = True
        colour = palette[i % len(palette)]
        ax.hist(vals, bins=bins, alpha=0.45, color=colour, label=label, density=True)
        p95 = percentile(vals, 95.0)
        ax.axvline(p95, color=colour, linestyle="--", linewidth=1.2)
        ax.text(p95, ax.get_ylim()[1] * 0.9, f"p95={p95:.1f}",
                rotation=90, fontsize=7, color=colour, va="top")
    if not plotted:
        return None
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, out_path)
    return out_path


# ── CPU vs GPU grouped bars ─────────────────────────────────────────────────
def plot_cpu_gpu_comparison(
    rows: List[Dict],
    out_path: Path,
    metric_field: str = "mean_latency_ms",
    title: str = "CPU vs GPU comparison",
    ylabel: str = "Mean latency (ms)",
    group_field: str = "workload",
) -> Optional[Path]:
    """Grouped bars: x = workload, hue = device."""
    plt = _plt()
    if plt is None or not rows:
        return None
    groups: List[str] = []
    devices: List[str] = []
    for r in rows:
        g = str(r.get(group_field, "?"))
        d = str(r.get("device", "?"))
        if g not in groups:
            groups.append(g)
        if d not in devices:
            devices.append(d)
    by_pair: Dict[Tuple[str, str], float] = {}
    for r in rows:
        v = r.get(metric_field)
        if v is None:
            continue
        by_pair[(str(r.get(group_field, "?")), str(r.get("device", "?")))] = float(v)
    if not by_pair:
        return None

    fig, ax = plt.subplots(figsize=(max(7, len(groups) * 1.3), 5.5))
    n_dev = max(1, len(devices))
    width = 0.8 / n_dev
    x_positions = list(range(len(groups)))
    for i, device in enumerate(devices):
        ys = [by_pair.get((g, device), 0.0) for g in groups]
        offsets = [x + (i - (n_dev - 1) / 2) * width for x in x_positions]
        ax.bar(offsets, ys, width=width, label=device,
               color=colour_for_device(device), alpha=0.85)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, out_path)
    return out_path


# ── Concurrency scaling ─────────────────────────────────────────────────────
def plot_concurrency_scaling(
    rows: List[Dict],
    out_path: Path,
    title: str = "Concurrency scaling",
) -> Optional[Path]:
    """Throughput and p95 latency vs concurrency, dual axis."""
    plt = _plt()
    if plt is None or not rows:
        return None
    by_label: Dict[str, List[Tuple[int, float, float]]] = {}
    for r in rows:
        c = int(r.get("concurrency", 0))
        tput = r.get("throughput_qps")
        p95 = r.get("p95_ms")
        if tput is None:
            continue
        by_label.setdefault(str(r.get("workload", "workload")), []).append(
            (c, float(tput), float(p95) if p95 is not None else 0.0)
        )
    if not by_label:
        return None

    fig, ax_t = plt.subplots()
    ax_l = ax_t.twinx()
    palette = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for i, (label, pts) in enumerate(sorted(by_label.items())):
        pts.sort()
        colour = palette[i % len(palette)]
        xs = [p[0] for p in pts]
        ax_t.plot(xs, [p[1] for p in pts], marker="o", color=colour,
                  linewidth=2, label=f"{label} throughput")
        ax_l.plot(xs, [p[2] for p in pts], marker="s", color=colour,
                  linewidth=1.4, linestyle="--", alpha=0.7,
                  label=f"{label} p95")
    ax_t.set_xlabel("Concurrency (workers)")
    ax_t.set_ylabel("Throughput (tasks/s)")
    ax_l.set_ylabel("p95 latency (ms)")
    lines, labels = ax_t.get_legend_handles_labels()
    l2, lab2 = ax_l.get_legend_handles_labels()
    ax_t.legend(lines + l2, labels + lab2, loc="best", fontsize=8)
    ax_t.set_title(title)
    save_fig(fig, out_path)
    return out_path


# ── Cold vs warm ────────────────────────────────────────────────────────────
def plot_cold_vs_warm(
    rows: List[Dict],
    out_path: Path,
    title: str = "Cold start vs warm steady-state",
) -> Optional[Path]:
    """Grouped bars of cold-start vs warm latency per device."""
    plt = _plt()
    if plt is None or not rows:
        return None
    labels = [str(r.get("label", r.get("device", "?"))) for r in rows]
    cold = [float(r.get("cold_start_ms", 0.0) or 0.0) for r in rows]
    warm = [float(r.get("warm_mean_ms", 0.0) or 0.0) for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.1), 5.5))
    x = list(range(len(labels)))
    width = 0.38
    ax.bar([i - width / 2 for i in x], cold, width=width,
           label="cold start (build + first call)", color="#d62728", alpha=0.85)
    ax.bar([i + width / 2 for i in x], warm, width=width,
           label="warm steady-state", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    save_fig(fig, out_path)
    return out_path
