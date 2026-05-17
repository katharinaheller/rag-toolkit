"""Context Recall metric for retrieval evaluation.

    Recall = |relevant ∩ retrieved| / |relevant|

Examples without relevance labels contribute 0.0 with a metadata warning.
"""

from __future__ import annotations

from typing import List, Set

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _recall_for_prediction(pred: EvaluationPrediction) -> float:
    use_chunks = (
        pred.example.relevant_chunk_ids is not None
        and len(pred.example.relevant_chunk_ids) > 0
    )

    if use_chunks:
        relevant_ids: Set[str] = set(pred.example.relevant_chunk_ids)  # type: ignore[arg-type]
        retrieved_ids: Set[str] = {ctx.chunk_id for ctx in pred.retrieved_contexts}
    else:
        relevant_ids = set(pred.example.relevant_document_ids)
        retrieved_ids = {ctx.document_id for ctx in pred.retrieved_contexts}

    if not relevant_ids:
        return 0.0

    hits = len(relevant_ids & retrieved_ids)
    return hits / len(relevant_ids)


class ContextRecallMetric(BaseMetric):
    """Mean Context Recall over all predictions. Range: [0, 1]."""

    @property
    def name(self) -> str:
        return "context_recall"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "note": "No predictions provided."},
            )

        per_example: List[float] = []
        n_no_labels = 0
        for pred in predictions:
            if not pred.example.relevant_document_ids and (
                pred.example.relevant_chunk_ids is None
                or not pred.example.relevant_chunk_ids
            ):
                n_no_labels += 1
            per_example.append(_recall_for_prediction(pred))

        mean = sum(per_example) / len(per_example)

        metadata = {"n": len(predictions)}
        if n_no_labels > 0:
            metadata["n_without_labels"] = n_no_labels
            metadata["warning"] = (
                f"{n_no_labels} example(s) had no relevance labels; "
                "they contribute 0.0 to recall."
            )

        return MetricResult(
            metric_name=self.name,
            value=round(mean, 6),
            per_example=[round(v, 6) for v in per_example],
            metadata=metadata,
        )
