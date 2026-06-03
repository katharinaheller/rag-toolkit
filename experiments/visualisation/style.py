"""Shared matplotlib style for publication-grade figures.

Keeps colours, font sizes, layout and export behaviour consistent across every
suite so all plots in the final report look like part of the same document.

Design decisions for robust, clipping-free, thesis-grade output
---------------------------------------------------------------
* ``constrained_layout`` is the single, authoritative layout engine. Unlike
  ``tight_layout`` it correctly accounts for ``twinx`` secondary axes, outside
  legends, suptitles and long tick/axis labels, and it produces deterministic
  results in headless (Agg) and interactive/notebook backends alike.
* ``savefig.bbox`` is deliberately NOT set globally. Combining a global
  ``bbox_inches="tight"`` with ``constrained_layout`` makes matplotlib emit a
  warning and silently discard part of the computed layout, which is the root
  cause of the clipped titles / right y-axis / legends. Export padding is
  instead handled centrally and consistently in :func:`save_fig`.
* Every figure should be created with ``new_figure`` / ``new_subplots`` so the
  layout engine and margins are applied uniformly.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Fixed palette so the same retriever always gets the same colour across plots.
RETRIEVER_PALETTE: Dict[str, str] = {
    "tfidf": "#7f7f7f",
    "bm25": "#9467bd",
    "dense_gemma": "#1f77b4",
    "dense_bge": "#2ca02c",
    "hybrid_gemma": "#ff7f0e",
    "hybrid_bge": "#d62728",
}

QUERY_TYPE_PALETTE: Dict[str, str] = {
    "keyword": "#4c72b0",
    "semantic_paraphrase": "#dd8452",
    "ambiguous": "#55a467",
    "noisy": "#c44e52",
    "multi_hop": "#8172b3",
}

# Constrained-layout padding (in fractions of font size / inches). These give a
# stable, generous margin so nothing touches the figure edge even at high DPI.
_CL_PAD = 0.06          # padding between axes and figure edge (w/h), fraction
_CL_HPAD = 0.05
_CL_WPAD = 0.05

# Export padding (inches) added around the (already laid-out) figure.
_EXPORT_PAD_INCHES = 0.08


def setup_matplotlib() -> bool:
    """Apply the thesis style. Returns ``False`` if matplotlib is missing.

    Idempotent: safe to call before every plot. Applies a constrained-layout
    based style and intentionally leaves ``savefig.bbox`` unset so the export
    path in :func:`save_fig` stays in full control.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "matplotlib is not installed — plots will be skipped. "
            "Install with: pip install matplotlib"
        )
        return False

    plt.rcParams.update({
        # ── Figure / DPI ────────────────────────────────────────────────────
        "figure.figsize": (9.0, 6.0),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        # IMPORTANT: do NOT set "savefig.bbox" globally. tight-bbox combined
        # with constrained_layout clips twin axes / suptitles. Export padding
        # is handled in save_fig().
        "savefig.pad_inches": _EXPORT_PAD_INCHES,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",

        # ── Layout engine ──────────────────────────────────────────────────
        # constrained_layout is twinx-safe and deterministic across backends.
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": _CL_HPAD,
        "figure.constrained_layout.w_pad": _CL_WPAD,
        "figure.constrained_layout.hspace": 0.06,
        "figure.constrained_layout.wspace": 0.06,

        # ── Typography ─────────────────────────────────────────────────────
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titlepad": 10,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "axes.labelpad": 6,
        "figure.titlesize": 14,
        "figure.titleweight": "bold",

        # ── Axes / grid ────────────────────────────────────────────────────
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.30,
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,

        # ── Legend ─────────────────────────────────────────────────────────
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.handlelength": 1.8,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.0,

        # ── Ticks ──────────────────────────────────────────────────────────
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 1.9,
        "lines.markersize": 5,
    })
    return True


# ── Figure factories ────────────────────────────────────────────────────────
def new_figure(figsize: Optional[Tuple[float, float]] = None):
    """Create a constrained-layout figure with project defaults.

    Returns the matplotlib Figure (or ``None`` if matplotlib is unavailable).
    """
    if not setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    fig = plt.figure(
        figsize=figsize,
        constrained_layout=True,
    )
    try:
        fig.get_layout_engine().set(h_pad=_CL_HPAD, w_pad=_CL_WPAD)
    except Exception:
        pass
    return fig


def new_subplots(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs,
):
    """Create constrained-layout subplots. Returns ``(fig, axes)`` or ``None``."""
    if not setup_matplotlib():
        return None
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        constrained_layout=True,
        **kwargs,
    )
    try:
        fig.get_layout_engine().set(h_pad=_CL_HPAD, w_pad=_CL_WPAD)
    except Exception:
        pass
    return fig, axes


def wrap_title(text: str, width: int = 60) -> str:
    """Wrap a long title (e.g. with a long GPU name) onto multiple lines.

    Keeps the figure width stable regardless of hardware-name length, which is
    what otherwise pushes a single-line title past the figure edge.
    """
    if not text:
        return text
    return "\n".join(textwrap.wrap(text, width=width)) or text


def colour_for_retriever(key: str) -> str:
    return RETRIEVER_PALETTE.get(key, "#444444")


def colour_for_query_type(key: str) -> str:
    return QUERY_TYPE_PALETTE.get(key, "#444444")


def place_outside_legend(ax, handles=None, labels=None, ncol: int = 2):
    """Place a legend *below* the axes so it never overlaps curves or titles.

    Works for single and twin-axes plots. Anchored to the axes (not the figure)
    so constrained_layout reserves the right amount of vertical space for it.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    n = len(labels)
    ncol = max(1, min(ncol, n)) if n else 1
    leg = ax.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=ncol,
        frameon=False,
        fontsize=9,
        borderaxespad=0.0,
    )
    return leg


def save_fig(fig, path: Path) -> None:
    """Save a figure with clipping-free, deterministic export settings.

    Layout strategy:
      * If the figure uses constrained_layout (the project default), we let the
        layout engine do its job and export WITHOUT ``bbox_inches="tight"``
        (which would fight the engine and clip twin axes / suptitles). A small
        uniform ``pad_inches`` guarantees breathing room at every edge.
      * If for some reason constrained_layout is not active, we fall back to a
        guarded ``tight_layout`` + ``bbox_inches="tight"`` path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    uses_constrained = False
    try:
        engine = fig.get_layout_engine()
        uses_constrained = (engine is not None) and (
            type(engine).__name__.lower().startswith("constrained")
        )
    except Exception:
        # Older matplotlib: probe the rcParam fallback.
        try:
            import matplotlib as mpl
            uses_constrained = bool(
                mpl.rcParams.get("figure.constrained_layout.use", False)
            )
        except Exception:
            uses_constrained = False

    try:
        if uses_constrained:
            # Do NOT pass bbox_inches="tight": it conflicts with the engine.
            fig.savefig(path, pad_inches=_EXPORT_PAD_INCHES)
        else:
            try:
                fig.tight_layout()
            except Exception:
                pass
            fig.savefig(path, bbox_inches="tight", pad_inches=_EXPORT_PAD_INCHES)
    except Exception as exc:  # last-resort robustness
        logger.warning("save_fig: primary export failed for %s (%s); retrying "
                       "with defensive settings.", path, exc)
        try:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
        except Exception as exc2:
            logger.error("save_fig: export failed for %s: %s", path, exc2)
    finally:
        try:
            fig.clear()
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass