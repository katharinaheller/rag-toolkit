"""Shared matplotlib style for publication-grade figures.

Keeps colours, font sizes and grid behaviour consistent across suites so
every plot in the final report looks like part of the same document.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

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


def setup_matplotlib() -> bool:
    """Apply the thesis style. Returns ``False`` if matplotlib is missing."""
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
        "figure.figsize": (8.5, 5.5),
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    return True


def colour_for_retriever(key: str) -> str:
    return RETRIEVER_PALETTE.get(key, "#444444")


def colour_for_query_type(key: str) -> str:
    return QUERY_TYPE_PALETTE.get(key, "#444444")


def save_fig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    try:
        fig.clear()
        import matplotlib.pyplot as plt
        plt.close(fig)
    except Exception:
        pass
