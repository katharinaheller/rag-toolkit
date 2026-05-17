"""Latency metric: per-stage wall-clock time in milliseconds.

Primary value is mean end-to-end latency. Per-stage means are in metadata.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _mean_or_none(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


class LatencyMetric(BaseMetric):
    """Per-stage and end-to-end latency statistics across all predictions."""

    @property
    def name(self) -> str:
        return "latency"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "note": "No predictions provided."},
            )

        retrieval_vals: List[float] = []
        reranking_vals: List[float] = []
        prompt_vals: List[float] = []
        generation_vals: List[float] = []
        e2e_vals: List[float] = []

        for pred in predictions:
            t = pred.timings
            if t.retrieval_ms is not None:
                retrieval_vals.append(t.retrieval_ms)
            if t.reranking_ms is not None:
                reranking_vals.append(t.reranking_ms)
            if t.prompt_construction_ms is not None:
                prompt_vals.append(t.prompt_construction_ms)
            if t.generation_ms is not None:
                generation_vals.append(t.generation_ms)
            if t.end_to_end_ms is not None:
                e2e_vals.append(t.end_to_end_ms)
            elif pred.generated_answer is not None:
                # Fallback: use generation latency from GenerationResult.
                e2e_vals.append(pred.generated_answer.latency_ms)

        mean_e2e = _mean_or_none(e2e_vals) or 0.0

        return MetricResult(
            metric_name=self.name,
            value=mean_e2e,
            per_example=e2e_vals if e2e_vals else None,
            metadata={
                "n": len(predictions),
                "unit": "milliseconds",
                "mean_retrieval_ms": _mean_or_none(retrieval_vals),
                "mean_reranking_ms": _mean_or_none(reranking_vals),
                "mean_prompt_construction_ms": _mean_or_none(prompt_vals),
                "mean_generation_ms": _mean_or_none(generation_vals),
                "mean_end_to_end_ms": _mean_or_none(e2e_vals),
                "n_e2e_measured": len(e2e_vals),
            },
        )
