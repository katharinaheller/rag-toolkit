"""Context Precision metric for retrieval evaluation.

    Precision@k = |relevant ∩ retrieved[:k]| / |retrieved[:k]|

Chunk-level matching is preferred when `relevant_chunk_ids` is provided.
Duplicates are scored on first occurrence only.
"""

from __future__ import annotations

from typing import List, Set

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _precision_for_prediction(pred: EvaluationPrediction) -> float:
    retrieved = pred.retrieved_contexts
    if not retrieved:
        return 0.0

    use_chunks = (
        pred.example.relevant_chunk_ids is not None
        and len(pred.example.relevant_chunk_ids) > 0
    )

    if use_chunks:
        relevant_ids: Set[str] = set(pred.example.relevant_chunk_ids)  # type: ignore[arg-type]
        seen: Set[str] = set()
        hits = 0
        for ctx in retrieved:
            cid = ctx.chunk_id
            if cid in seen:
                continue
            seen.add(cid)
            if cid in relevant_ids:
                hits += 1
    else:
        relevant_ids = set(pred.example.relevant_document_ids)
        seen = set()
        hits = 0
        for ctx in retrieved:
            did = ctx.document_id
            if did in seen:
                continue
            seen.add(did)
            if did in relevant_ids:
                hits += 1

    unique_retrieved = len(seen)
    return hits / unique_retrieved if unique_retrieved > 0 else 0.0


class ContextPrecisionMetric(BaseMetric):
    """Mean Context Precision over all predictions. Range: [0, 1]."""

    @property
    def name(self) -> str:
        return "context_precision"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "note": "No predictions provided."},
            )

        per_example: List[float] = [
            _precision_for_prediction(p) for p in predictions
        ]
        mean = sum(per_example) / len(per_example)

        return MetricResult(
            metric_name=self.name,
            value=round(mean, 6),
            per_example=[round(v, 6) for v in per_example],
            metadata={"n": len(predictions)},
        )
