"""Mean Reciprocal Rank.

    RR(q) = 1 / rank_of_first_relevant_result    (0.0 if none found)
    MRR   = mean(RR(q)) over all queries
"""

from __future__ import annotations

from typing import List, Set

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _reciprocal_rank(pred: EvaluationPrediction) -> float:
    retrieved = sorted(pred.retrieved_contexts, key=lambda c: c.rank)

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
    position = 0
    for ctx in retrieved:
        key = ctx.chunk_id if use_chunks else ctx.document_id
        if key in seen:
            continue
        seen.add(key)
        position += 1
        if key in relevant_ids:
            return 1.0 / position

    return 0.0


class MRRMetric(BaseMetric):
    """Average reciprocal rank over all predictions. Range: [0, 1]."""

    @property
    def name(self) -> str:
        return "mrr"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "note": "No predictions provided."},
            )

        per_example: List[float] = [_reciprocal_rank(p) for p in predictions]
        mean = sum(per_example) / len(per_example)

        return MetricResult(
            metric_name=self.name,
            value=round(mean, 6),
            per_example=[round(v, 6) for v in per_example],
            metadata={"n": len(predictions)},
        )
