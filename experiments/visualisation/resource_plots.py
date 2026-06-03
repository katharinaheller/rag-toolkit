"""HPC-style resource visualisation.

Timeline curves (CPU%, RAM, GPU%, VRAM vs wall clock), GPU utilisation
heatmaps, and CPU-vs-GPU speedup bars. Every function returns the output path
on success or ``None`` when matplotlib is unavailable, mirroring the
conventions in ``experiments.visualisation.plots``.

All figures use the project's constrained-layout factories and the
clipping-free :func:`save_fig` export, so titles, twin (right) y-axes and
outside legends are never cut off — including for long GPU names and high DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.visualisation.style import (
    colour_for_retriever,
    new_figure,
    new_subplots,
    place_outside_legend,
    save_fig,
    setup_matplotlib,
    wrap_title,
)

logger = logging.getLogger(__name__)

# Stable colours for device families so CPU/GPU look the same everywhere.
DEVICE_PALETTE: Dict[str, str] = {
    "cpu": "#1f77b4",
    "gpu": "#d62728",
    "cuda": "#d62728",
    "cuda:0": "#d62728",
    "cuda:1": "#e377c2",
}


def colour_for_device(key: str) -> str:
    key = (key or "").lower()
    if key in DEVICE_PALETTE:
        return DEVICE_PALETTE[key]
    if key.startswith("cuda") or key.startswith("gpu"):
        return DEVICE_PALETTE["gpu"]
    return DEVICE_PALETTE["cpu"]


def _plt():
    if not setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    return plt


# ── Timeline curves ─────────────────────────────────────────────────────────
def plot_resource_timeline(
    samples: List[Dict],
    out_path: Path,
    title: str = "Resource utilisation over time",
) -> Optional[Path]:
    """Plot CPU%, GPU%, RSS, VRAM against wall-clock seconds.

    ``samples`` rows are :class:`ResourceSample` dicts. Missing axes (e.g. no
    GPU) are simply omitted; the function never raises on partial data.

    Layout robustness
    -----------------
    * constrained_layout sizes the axes so the suptitle, the right-hand memory
      axis label and the outside legend all fit without clipping.
    * The legend is placed *below* the plot (anchored to the utilisation axis),
      so curves near the top of the panel never collide with it and the upper
      plot region is fully visible.
    * The title is wrapped so a long GPU name cannot push text past the edge.
    """
    if _plt() is None or not samples:
        return None

    ts = [float(s.get("t_s", 0.0)) for s in samples]

    def _col(key: str) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for s in samples:
            v = s.get(key)
            out.append(float(v) if v is not None else None)
        return out

    cpu = _col("cpu_percent")
    gpu = _col("gpu_utilisation_percent")
    rss = _col("memory_rss_mb")
    vram = _col("gpu_memory_used_mb")
    torch_alloc = _col("gpu_torch_allocated_mb")

    have_gpu_util = any(v is not None for v in gpu)
    have_vram = any(v is not None for v in vram)
    have_torch_alloc = any(v is not None for v in torch_alloc)

    # A touch taller than wide to leave room for the below-axes legend and an
    # optionally two-line (wrapped) title.
    created = new_subplots(figsize=(10.0, 6.2))
    if created is None:
        return None
    fig, ax_util = created
    ax_mem = ax_util.twinx()

    # Utilisation axis (percent).
    _plot_clean(ax_util, ts, cpu, label="CPU %", colour="#1f77b4")
    if have_gpu_util:
        _plot_clean(ax_util, ts, gpu, label="GPU %", colour="#d62728")
    ax_util.set_xlabel("Wall-clock time (s)")
    ax_util.set_ylabel("Utilisation (%)")
    ax_util.set_ylim(0, 105)
    ax_util.margins(x=0.01)

    # Memory axis (MiB) on the right twin.
    _plot_clean(ax_mem, ts, rss, label="RSS (MiB)", colour="#2ca02c",
                linestyle="--")
    if have_vram:
        _plot_clean(ax_mem, ts, vram, label="VRAM used (MiB)",
                    colour="#ff7f0e", linestyle="--")
    elif have_torch_alloc:
        _plot_clean(ax_mem, ts, torch_alloc, label="VRAM alloc (MiB)",
                    colour="#ff7f0e", linestyle="--")
    ax_mem.set_ylabel("Memory (MiB)")
    # The twin axis owns the right spine; make it visible (global style hides
    # the right spine, which would otherwise drop the right axis line).
    ax_mem.spines["right"].set_visible(True)
    ax_mem.spines["top"].set_visible(False)
    ax_mem.grid(False)  # avoid double grid from the twin

    # Combined legend BELOW the axes (never overlaps the upper plot / title).
    lines, labels = ax_util.get_legend_handles_labels()
    l2, lab2 = ax_mem.get_legend_handles_labels()
    all_lines = lines + l2
    all_labels = labels + lab2
    place_outside_legend(ax_util, all_lines, all_labels,
                         ncol=min(4, max(1, len(all_labels))))

    ax_util.set_title(wrap_title(title, width=58))

    save_fig(fig, out_path)
    return out_path


def _plot_clean(ax, xs, ys, label, colour, linestyle="-") -> None:
    """Plot a series, skipping ``None`` gaps without breaking the line badly."""
    pair_x: List[float] = []
    pair_y: List[float] = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        pair_x.append(x)
        pair_y.append(y)
    if pair_x:
        ax.plot(pair_x, pair_y, label=label, color=colour,
                linewidth=1.8, linestyle=linestyle, alpha=0.92)


# ── GPU utilisation heatmap ─────────────────────────────────────────────────
def plot_gpu_utilisation_heatmap(
    timelines: Dict[str, List[Dict]],
    out_path: Path,
    title: str = "GPU utilisation heatmap",
    bins: int = 40,
) -> Optional[Path]:
    """Heatmap of GPU% over time, one row per labelled timeline.

    ``timelines`` maps a label (e.g. "bs=16") to its list of sample dicts.
    Each timeline is resampled into ``bins`` equal-width time buckets. If no
    timeline carries GPU data, returns ``None``.
    """
    plt = _plt()
    if plt is None or not timelines:
        return None

    labels = list(timelines.keys())
    grid: List[List[float]] = []
    any_gpu = False
    for label in labels:
        samples = timelines[label]
        row = _resample_series(samples, "gpu_utilisation_percent", bins)
        if any(v > 0 for v in row):
            any_gpu = True
        grid.append(row)

    if not any_gpu:
        return None

    created = new_subplots(
        figsize=(max(7.5, bins * 0.18), max(3.4, len(labels) * 0.7 + 1.4))
    )
    if created is None:
        return None
    fig, ax = created

    im = ax.imshow(grid, aspect="auto", cmap="inferno", vmin=0, vmax=100,
                   interpolation="nearest")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Normalised time ({bins} bins)")
    ax.set_title(wrap_title(title, width=58))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("GPU utilisation (%)")
    save_fig(fig, out_path)
    return out_path


def _resample_series(samples: List[Dict], key: str, bins: int) -> List[float]:
    """Bucket a sample series into ``bins`` mean values over normalised time."""
    if not samples:
        return [0.0] * bins
    ts = [float(s.get("t_s", 0.0)) for s in samples]
    t_max = max(ts) if ts else 0.0
    buckets: List[List[float]] = [[] for _ in range(bins)]
    for s in samples:
        v = s.get(key)
        if v is None:
            continue
        t = float(s.get("t_s", 0.0))
        idx = 0 if t_max <= 0 else min(bins - 1, int((t / t_max) * bins))
        buckets[idx].append(float(v))
    out: List[float] = []
    last = 0.0
    for b in buckets:
        if b:
            last = sum(b) / len(b)
        out.append(last)  # forward-fill empty buckets
    return out


# ── Speedup bars ────────────────────────────────────────────────────────────
def plot_speedup_bars(
    rows: List[Dict],
    out_path: Path,
    title: str = "GPU speedup over CPU",
    label_field: str = "label",
    speedup_field: str = "speedup",
) -> Optional[Path]:
    """Bar chart of speedup factor per workload (1.0 line = parity)."""
    plt = _plt()
    if plt is None or not rows:
        return None
    labels = [str(r.get(label_field, "?")) for r in rows]
    values = [float(r.get(speedup_field, 0.0)) for r in rows]

    created = new_subplots(figsize=(max(6.5, len(labels) * 0.95), 5.2))
    if created is None:
        return None
    fig, ax = created

    colours = ["#2ca02c" if v >= 1.0 else "#d62728" for v in values]
    bars = ax.bar(labels, values, color=colours, alpha=0.85)
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0,
               label="parity (CPU = GPU)")
    # Headroom so the value labels above the tallest bar are not clipped.
    if values:
        ax.set_ylim(0, max(values) * 1.18 + 0.05)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}×",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Speedup factor (CPU time / GPU time)")
    ax.set_title(wrap_title(title, width=58))
    place_outside_legend(ax, ncol=1)
    ax.tick_params(axis="x", rotation=20)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    save_fig(fig, out_path)
    return out_path


# ── VRAM vs batch size ──────────────────────────────────────────────────────
def plot_vram_vs_batchsize(
    rows: List[Dict],
    out_path: Path,
    title: str = "Peak VRAM vs batch size",
) -> Optional[Path]:
    """Line plot of peak VRAM (MiB) against batch size, one line per device."""
    plt = _plt()
    if plt is None or not rows:
        return None
    by_device: Dict[str, List[Tuple[int, float]]] = {}
    for r in rows:
        vram = r.get("gpu_peak_memory_mb")
        if vram is None:
            continue
        bs = int(r.get("batch_size", 0))
        by_device.setdefault(str(r.get("device", "?")), []).append((bs, float(vram)))
    if not by_device:
        return None

    created = new_subplots(figsize=(9.0, 5.6))
    if created is None:
        return None
    fig, ax = created

    for device, pts in sorted(by_device.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                marker="o", label=device, color=colour_for_device(device),
                linewidth=2)
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Peak VRAM (MiB)")
    ax.set_title(wrap_title(title, width=58))
    place_outside_legend(ax, ncol=min(4, len(by_device)))
    save_fig(fig, out_path)
    return out_path