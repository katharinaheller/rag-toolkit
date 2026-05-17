"""Throughput metric: examples processed per second during the evaluation run.

    Throughput (examples/s) = n_examples / total_duration_s
"""

from __future__ import annotations

from typing import List, Optional

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


class ThroughputMetric(BaseMetric):
    """Requires total_duration_s injected by the EvaluationRunner."""

    def __init__(self, total_duration_s: Optional[float] = None) -> None:
        self._total_duration_s = total_duration_s

    @property
    def name(self) -> str:
        return "throughput"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        n = len(predictions)

        if n == 0 or not self._total_duration_s or self._total_duration_s <= 0.0:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=None,
                metadata={
                    "n": n,
                    "unit": "examples_per_second",
                    "total_duration_s": self._total_duration_s,
                    "note": "Throughput requires n > 0 and total_duration_s > 0.",
                },
            )

        throughput = n / self._total_duration_s

        return MetricResult(
            metric_name=self.name,
            value=round(throughput, 4),
            per_example=None,
            metadata={
                "n": n,
                "unit": "examples_per_second",
                "total_duration_s": round(self._total_duration_s, 3),
            },
        )
