"""All matplotlib figures used by the experiment suites and the final report.

Every plot function returns the absolute output path so callers can stitch it
into the report. If matplotlib is unavailable, ``setup_matplotlib`` returns
False and these functions become no-ops that return ``None``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.visualisation.style import (
    QUERY_TYPE_PALETTE,
    colour_for_query_type,
    colour_for_retriever,
    save_fig,
    setup_matplotlib,
)

logger = logging.getLogger(__name__)


# ── Helper: lazy mpl import that respects setup_matplotlib() failure ────────
def _plt():
    if not setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    return plt


# ── Scaling curves ──────────────────────────────────────────────────────────
def plot_retriever_corpus_scaling(
    data: List[Dict],
    metric: str,
    out_path: Path,
    title: str,
) -> Optional[Path]:
    """Plot ``metric`` vs corpus size, one line per retriever.

    ``data`` rows must contain: retriever, corpus_size, metric.
    """
    plt = _plt()
    if plt is None:
        return None

    by_retriever: Dict[str, List[Tuple[int, float]]] = {}
    for row in data:
        by_retriever.setdefault(row["retriever"], []).append(
            (int(row["corpus_size"]), float(row[metric]))
        )

    fig, ax = plt.subplots()
    for retriever, pts in sorted(by_retriever.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys, marker="o", label=retriever,
            color=colour_for_retriever(retriever), linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Corpus size (documents, log scale)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    save_fig(fig, out_path)
    return out_path


# ── Top-k sensitivity ───────────────────────────────────────────────────────
def plot_topk_sensitivity(
    data: List[Dict],
    metric: str,
    out_path: Path,
    title: str,
) -> Optional[Path]:
    """Plot ``metric`` vs top_k, one line per retriever."""
    plt = _plt()
    if plt is None:
        return None
    by_retriever: Dict[str, List[Tuple[int, float]]] = {}
    for row in data:
        by_retriever.setdefault(row["retriever"], []).append(
            (int(row["top_k"]), float(row[metric]))
        )

    fig, ax = plt.subplots()
    for retriever, pts in sorted(by_retriever.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=retriever,
                color=colour_for_retriever(retriever), linewidth=2)
    ax.set_xlabel("top_k")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    save_fig(fig, out_path)
    return out_path


# ── Grouped query-type bar chart ────────────────────────────────────────────
def plot_query_type_grouped_bars(
    data: List[Dict],
    metric: str,
    out_path: Path,
    title: str,
) -> Optional[Path]:
    """Grouped bar chart: x = query_type, hue = retriever."""
    plt = _plt()
    if plt is None:
        return None

    query_types: List[str] = []
    for row in data:
        if row["query_type"] not in query_types:
            query_types.append(row["query_type"])
    retrievers: List[str] = []
    for row in data:
        if row["retriever"] not in retrievers:
            retrievers.append(row["retriever"])

    by_pair: Dict[Tuple[str, str], float] = {
        (row["query_type"], row["retriever"]): float(row[metric]) for row in data
    }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    n_retrievers = max(1, len(retrievers))
    bar_width = 0.8 / n_retrievers
    x_positions = list(range(len(query_types)))

    for i, retr in enumerate(retrievers):
        ys = [by_pair.get((qt, retr), 0.0) for qt in query_types]
        offsets = [x + (i - (n_retrievers - 1) / 2) * bar_width for x in x_positions]
        ax.bar(offsets, ys, width=bar_width,
               label=retr, color=colour_for_retriever(retr))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(query_types, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    save_fig(fig, out_path)
    return out_path


# ── Heatmap (overlap matrix etc) ───────────────────────────────────────────
def plot_heatmap(
    matrix: Dict[str, Dict[str, float]],
    out_path: Path,
    title: str,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Optional[Path]:
    plt = _plt()
    if plt is None:
        return None

    rows = sorted(matrix.keys())
    cols = sorted({c for row in matrix.values() for c in row.keys()})
    grid = [[matrix[r].get(c, 0.0) for c in cols] for r in rows]

    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 1.0), max(5, len(rows) * 0.7)))
    im = ax.imshow(grid, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            val = grid[i][j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    ax.set_title(title)
    save_fig(fig, out_path)
    return out_path


# ── Pareto plot ─────────────────────────────────────────────────────────────
def plot_pareto(
    points: List[Dict],
    front_labels: List[str],
    out_path: Path,
    x_field: str,
    y_field: str,
    title: str,
    x_label: str,
    y_label: str,
    label_field: str = "label",
) -> Optional[Path]:
    plt = _plt()
    if plt is None:
        return None

    fig, ax = plt.subplots()
    xs = [float(p[x_field]) for p in points]
    ys = [float(p[y_field]) for p in points]
    labels = [str(p[label_field]) for p in points]

    for x, y, label in zip(xs, ys, labels):
        is_front = label in front_labels
        ax.scatter(x, y, s=80 if is_front else 40,
                   color="#d62728" if is_front else "#888888",
                   edgecolors="black" if is_front else "none",
                   zorder=3 if is_front else 2,
                   label="Pareto-optimal" if is_front and "Pareto-optimal" not in ax.get_legend_handles_labels()[1] else None)
        ax.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    # Draw front line.
    front_points = [(float(p[x_field]), float(p[y_field]), str(p[label_field]))
                    for p in points if str(p[label_field]) in front_labels]
    front_points.sort()
    if len(front_points) > 1:
        ax.plot([p[0] for p in front_points], [p[1] for p in front_points],
                linestyle="--", color="#d62728", alpha=0.6, zorder=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(loc="best")
    save_fig(fig, out_path)
    return out_path


# ── Scatter / Box plots ─────────────────────────────────────────────────────
def plot_latency_quality_scatter(
    points: List[Dict],
    out_path: Path,
    title: str,
    x_field: str = "latency_ms",
    y_field: str = "recall",
) -> Optional[Path]:
    plt = _plt()
    if plt is None:
        return None
    fig, ax = plt.subplots()
    by_retriever: Dict[str, List[Tuple[float, float]]] = {}
    for p in points:
        by_retriever.setdefault(p["retriever"], []).append(
            (float(p[x_field]), float(p[y_field]))
        )
    for retr, vals in sorted(by_retriever.items()):
        xs = [v[0] for v in vals]
        ys = [v[1] for v in vals]
        ax.scatter(xs, ys, label=retr, color=colour_for_retriever(retr),
                   alpha=0.7, s=42)
    ax.set_xlabel(x_field)
    ax.set_ylabel(y_field)
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    save_fig(fig, out_path)
    return out_path


def plot_box_by_group(
    grouped_values: Dict[str, List[float]],
    out_path: Path,
    title: str,
    ylabel: str,
) -> Optional[Path]:
    plt = _plt()
    if plt is None:
        return None
    groups = sorted(grouped_values.keys())
    data = [grouped_values[g] for g in groups]
    fig, ax = plt.subplots()
    bp = ax.boxplot(data, labels=groups, patch_artist=True, showfliers=False)
    for patch, group in zip(bp["boxes"], groups):
        patch.set_facecolor(colour_for_retriever(group))
        patch.set_alpha(0.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, out_path)
    return out_path


# ── Stability variance / error bars ─────────────────────────────────────────
def plot_stability_bars(
    data: List[Dict],
    metric: str,
    out_path: Path,
    title: str,
) -> Optional[Path]:
    plt = _plt()
    if plt is None:
        return None
    retrievers = sorted({r["retriever"] for r in data})
    means = []
    lows = []
    highs = []
    for r in retrievers:
        rows = [row for row in data if row["retriever"] == r]
        m = sum(float(row[metric]) for row in rows) / max(1, len(rows))
        means.append(m)
        lo = min((float(row.get(f"{metric}_ci_lo", row[metric])) for row in rows), default=m)
        hi = max((float(row.get(f"{metric}_ci_hi", row[metric])) for row in rows), default=m)
        lows.append(m - lo)
        highs.append(hi - m)
    fig, ax = plt.subplots()
    ax.bar(retrievers, means, color=[colour_for_retriever(r) for r in retrievers],
           yerr=[lows, highs], capsize=5)
    ax.set_ylabel(metric)
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    save_fig(fig, out_path)
    return out_path


# ── Chunk relevance decay (rank vs hit-rate) ───────────────────────────────
def plot_rank_decay(
    data: List[Dict],
    out_path: Path,
    title: str,
) -> Optional[Path]:
    plt = _plt()
    if plt is None:
        return None
    by_retriever: Dict[str, Dict[int, float]] = {}
    for row in data:
        by_retriever.setdefault(row["retriever"], {})[int(row["rank"])] = float(row["hit_rate"])
    fig, ax = plt.subplots()
    for r, ranks in sorted(by_retriever.items()):
        xs = sorted(ranks.keys())
        ys = [ranks[k] for k in xs]
        ax.plot(xs, ys, marker="o", label=r, color=colour_for_retriever(r), linewidth=2)
    ax.set_xlabel("Rank position")
    ax.set_ylabel("Hit rate at rank")
    ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    save_fig(fig, out_path)
    return out_path


# ── Dashboard: 4-panel summary ─────────────────────────────────────────────
def plot_dashboard(
    summary: Dict[str, List[Dict]],
    out_path: Path,
) -> Optional[Path]:
    """Combine four headline panels in a single thesis-ready figure."""
    plt = _plt()
    if plt is None:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panels = ["scaling", "topk", "pareto", "stability"]
    titles = {
        "scaling": "Recall@10 vs corpus size",
        "topk": "Recall@k vs k (largest corpus)",
        "pareto": "Latency vs Recall@10",
        "stability": "Top-10 rank stability across reruns",
    }

    # Panel 1 — scaling.
    scaling_rows = summary.get("scaling", [])
    by_retriever: Dict[str, List[Tuple[int, float]]] = {}
    for r in scaling_rows:
        by_retriever.setdefault(r["retriever"], []).append(
            (int(r["corpus_size"]), float(r["recall@10"]))
        )
    ax = axes[0][0]
    for retriever, pts in sorted(by_retriever.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", label=retriever, color=colour_for_retriever(retriever))
    ax.set_xscale("log")
    ax.set_title(titles["scaling"])
    ax.set_xlabel("Corpus size")
    ax.set_ylabel("recall@10")
    ax.legend(fontsize=8, ncol=2)

    # Panel 2 — topk.
    topk_rows = summary.get("topk", [])
    by_retriever = {}
    for r in topk_rows:
        by_retriever.setdefault(r["retriever"], []).append(
            (int(r["top_k"]), float(r["recall"]))
        )
    ax = axes[0][1]
    for retriever, pts in sorted(by_retriever.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", label=retriever, color=colour_for_retriever(retriever))
    ax.set_title(titles["topk"])
    ax.set_xlabel("top_k")
    ax.set_ylabel("recall@k")
    ax.legend(fontsize=8, ncol=2)

    # Panel 3 — pareto.
    pareto_rows = summary.get("pareto", [])
    ax = axes[1][0]
    for r in pareto_rows:
        ax.scatter(float(r["latency_ms"]), float(r["recall"]),
                   color=colour_for_retriever(r["retriever"]),
                   s=60, alpha=0.85,
                   label=r["retriever"] if not any(child.get_label() == r["retriever"] for child in ax.collections[:-1]) else None)
        ax.annotate(r["label"], (float(r["latency_ms"]), float(r["recall"])),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_title(titles["pareto"])
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("recall@10")
    ax.legend(fontsize=8, ncol=2)

    # Panel 4 — stability.
    stab_rows = summary.get("stability", [])
    retrievers = sorted({r["retriever"] for r in stab_rows})
    means = []
    for retr in retrievers:
        vals = [float(r["stability"]) for r in stab_rows if r["retriever"] == retr]
        means.append(sum(vals) / len(vals) if vals else 0.0)
    ax = axes[1][1]
    ax.bar(retrievers, means, color=[colour_for_retriever(r) for r in retrievers])
    ax.set_title(titles["stability"])
    ax.set_ylabel("Jaccard@10 between reruns")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("RAG Experiment Dashboard", fontsize=15, y=1.01)
    fig.tight_layout()
    save_fig(fig, out_path)
    return out_path
