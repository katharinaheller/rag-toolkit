from experiments.visualisation.style import (
    QUERY_TYPE_PALETTE,
    RETRIEVER_PALETTE,
    colour_for_query_type,
    colour_for_retriever,
    save_fig,
    setup_matplotlib,
)
from experiments.visualisation.plots import (
    plot_box_by_group,
    plot_dashboard,
    plot_heatmap,
    plot_latency_quality_scatter,
    plot_pareto,
    plot_query_type_grouped_bars,
    plot_rank_decay,
    plot_retriever_corpus_scaling,
    plot_stability_bars,
    plot_topk_sensitivity,
)

__all__ = [
    "QUERY_TYPE_PALETTE",
    "RETRIEVER_PALETTE",
    "colour_for_query_type",
    "colour_for_retriever",
    "plot_box_by_group",
    "plot_dashboard",
    "plot_heatmap",
    "plot_latency_quality_scatter",
    "plot_pareto",
    "plot_query_type_grouped_bars",
    "plot_rank_decay",
    "plot_retriever_corpus_scaling",
    "plot_stability_bars",
    "plot_topk_sensitivity",
    "save_fig",
    "setup_matplotlib",
]
