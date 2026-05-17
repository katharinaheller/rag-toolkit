"""Abstract base class for all evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag.evaluation.types import EvaluationPrediction, MetricResult


class BaseMetric(ABC):
    """Contract for all RAG evaluation metrics.

    Implementations must be stateless: evaluate() produces the same result for
    the same inputs every call, with no side effects.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short stable identifier (e.g. 'context_precision')."""

    @abstractmethod
    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        """Compute the metric over a list of predictions. Empty input is valid."""
