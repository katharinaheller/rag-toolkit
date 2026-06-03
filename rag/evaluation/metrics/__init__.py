"""rag.evaluation.metrics - the single home for all RAG evaluation metrics.

This package exposes two complementary layers:

* Prediction-level metric classes (``BaseMetric`` subclasses) that operate on
  :class:`~rag.evaluation.types.EvaluationPrediction` objects and are wired
  into the :class:`~rag.evaluation.runner.EvaluationRunner`.
* Functional, id/string-level metric primitives (in :mod:`.functional`) used by
  the experiment suites.

Experiments and other callers import metrics exclusively from here.
"""

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.metrics.context_precision import ContextPrecisionMetric
from rag.evaluation.metrics.context_recall import ContextRecallMetric
from rag.evaluation.metrics.exact_match import ExactMatchMetric
from rag.evaluation.metrics.latency import LatencyMetric
from rag.evaluation.metrics.memory_usage import MemoryUsageMetric
from rag.evaluation.metrics.mrr import MRRMetric
from rag.evaluation.metrics.ndcg import NDCGMetric
from rag.evaluation.metrics.throughput import ThroughputMetric
from rag.evaluation.metrics.token_f1 import TokenF1Metric
from rag.evaluation.metrics.functional import (
    ParetoPoint,
    aggregate_retrieval_metrics,
    bootstrap_ci,
    coefficient_of_variation,
    compute_pareto_front,
    context_overlap,
    context_pollution_ratio,
    dcg,
    hallucination_score,
    hit_at_k,
    jaccard,
    mean,
    ndcg_at_k,
    overlap_at_k,
    pairwise_overlap_matrix,
    precision_at_k,
    rank_stability,
    recall_at_k,
    reciprocal_rank,
    std_dev,
)

__all__ = [
    # Prediction-level metric classes
    "BaseMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "ExactMatchMetric",
    "LatencyMetric",
    "MemoryUsageMetric",
    "MRRMetric",
    "NDCGMetric",
    "ThroughputMetric",
    "TokenF1Metric",
    # Functional metric primitives
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
