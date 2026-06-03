"""Normalised Discounted Cumulative Gain.

Binary relevance.

    DCG@k  = Σ_{i=1}^{k}  rel_i / log2(i + 1)
    IDCG@k = DCG of the ideal ranking
    nDCG@k = DCG@k / IDCG@k    (0.0 when IDCG = 0)
"""

from __future__ import annotations

from typing import List, Optional, Set

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.metrics.functional.retrieval import dcg as _dcg
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _ndcg_for_prediction(pred: EvaluationPrediction, k: Optional[int]) -> float:
    retrieved = sorted(pred.retrieved_contexts, key=lambda c: c.rank)
    if k is not None:
        retrieved = retrieved[:k]

    use_chunks = (
        pred.example.relevant_chunk_ids is not None
        and len(pred.example.relevant_chunk_ids) > 0
    )

    if use_chunks:
        relevant_ids: Set[str] = set(pred.example.relevant_chunk_ids)  # type: ignore[arg-type]
    else:
        relevant_ids = set(pred.example.relevant_document_ids)

    if not relevant_ids:
        return 0.0

    seen: Set[str] = set()
    relevances: List[int] = []
    for ctx in retrieved:
        key = ctx.chunk_id if use_chunks else ctx.document_id
        if key in seen:
            relevances.append(0)  # Duplicates count as not relevant.
            continue
        seen.add(key)
        relevances.append(1 if key in relevant_ids else 0)

    dcg = _dcg(relevances)

    n_relevant = min(len(relevant_ids), len(relevances))
    ideal_relevances = [1] * n_relevant + [0] * (len(relevances) - n_relevant)
    idcg = _dcg(ideal_relevances)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


class NDCGMetric(BaseMetric):
    """Mean nDCG over all predictions. Always report results with the k used."""

    def __init__(self, k: Optional[int] = None) -> None:
        if k is not None and k <= 0:
            raise ValueError(f"NDCGMetric: k must be positive or None, got {k}.")
        self._k = k

    @property
    def name(self) -> str:
        suffix = f"@{self._k}" if self._k is not None else ""
        return f"ndcg{suffix}"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "k": self._k, "note": "No predictions provided."},
            )

        per_example: List[float] = [
            _ndcg_for_prediction(p, self._k) for p in predictions
        ]
        mean = sum(per_example) / len(per_example)

        return MetricResult(
            metric_name=self.name,
            value=round(mean, 6),
            per_example=[round(v, 6) for v in per_example],
            metadata={"n": len(predictions), "k": self._k},
        )
