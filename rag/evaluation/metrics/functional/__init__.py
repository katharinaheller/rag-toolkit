"""Functional, id/string-level evaluation metrics.

These are the low-level metric primitives (retrieval IR metrics, overlap and
rank-correlation, stability/bootstrap, faithfulness proxies, Pareto frontier)
used by the experiment suites. They live here, alongside the prediction-level
``BaseMetric`` classes, so that every metric in the project has a single home
in the evaluation module.
"""

from rag.evaluation.metrics.functional.faithfulness import (
    context_overlap,
    context_pollution_ratio,
    hallucination_score,
)
from rag.evaluation.metrics.functional.overlap import (
    jaccard,
    overlap_at_k,
    pairwise_overlap_matrix,
)
from rag.evaluation.metrics.functional.pareto import ParetoPoint, compute_pareto_front
from rag.evaluation.metrics.functional.retrieval import (
    aggregate_retrieval_metrics,
    dcg,
    hit_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.evaluation.metrics.functional.stability import (
    bootstrap_ci,
    coefficient_of_variation,
    rank_stability,
    std_dev,
)

__all__ = [
    "ParetoPoint",
    "aggregate_retrieval_metrics",
    "bootstrap_ci",
    "coefficient_of_variation",
    "compute_pareto_front",
    "context_overlap",
    "context_pollution_ratio",
    "dcg",
    "hallucination_score",
    "hit_at_k",
    "jaccard",
    "mean",
    "ndcg_at_k",
    "overlap_at_k",
    "pairwise_overlap_matrix",
    "precision_at_k",
    "rank_stability",
    "recall_at_k",
    "reciprocal_rank",
    "std_dev",
]
