"""Memory usage metric: peak and mean RSS in MB across ResourceSnapshots."""

from __future__ import annotations

from typing import List, Optional

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


class MemoryUsageMetric(BaseMetric):
    """Primary value: peak RSS across all collected snapshots. Mean and min in metadata."""

    @property
    def name(self) -> str:
        return "memory_usage"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                metadata={"n": 0, "unit": "MB", "note": "No predictions provided."},
            )

        rss_values: List[float] = []
        for pred in predictions:
            for snap in pred.resource_snapshots:
                if snap.memory_rss_mb is not None:
                    rss_values.append(snap.memory_rss_mb)

        if not rss_values:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                metadata={
                    "n": len(predictions),
                    "unit": "MB",
                    "n_snapshots": 0,
                    "note": (
                        "No memory snapshots available. "
                        "Set capture_resources=True in EvaluationConfig."
                    ),
                },
            )

        peak = max(rss_values)
        mean = sum(rss_values) / len(rss_values)

        return MetricResult(
            metric_name=self.name,
            value=round(peak, 2),
            per_example=None,
            metadata={
                "n": len(predictions),
                "n_snapshots": len(rss_values),
                "unit": "MB",
                "peak_rss_mb": round(peak, 2),
                "mean_rss_mb": round(mean, 2),
                "min_rss_mb": round(min(rss_values), 2),
            },
        )
