"""Exact Match metric for generation evaluation.

Strictest answer-quality metric; most useful for short factoid answers.
"""

from __future__ import annotations

import re
from typing import List, Optional

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _normalise(text: str, *, case_sensitive: bool, strip_punctuation: bool) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if strip_punctuation:
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    if not case_sensitive:
        text = text.lower()
    return text


class ExactMatchMetric(BaseMetric):
    """Proportion of predictions that exactly match the expected answer.

        EM = (number of exact matches) / (number of examples)
    """

    def __init__(
        self,
        case_sensitive: bool = False,
        strip_punctuation: bool = False,
    ) -> None:
        self._case_sensitive = case_sensitive
        self._strip_punctuation = strip_punctuation

    @property
    def name(self) -> str:
        return "exact_match"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "note": "No predictions provided."},
            )

        per_example: List[float] = []
        for pred in predictions:
            generated_text = (
                pred.generated_answer.text if pred.generated_answer is not None else ""
            )
            expected = pred.example.expected_answer
            match = self._is_match(generated_text, expected)
            per_example.append(1.0 if match else 0.0)

        aggregate = sum(per_example) / len(per_example)

        return MetricResult(
            metric_name=self.name,
            value=round(aggregate, 6),
            per_example=per_example,
            metadata={
                "n": len(predictions),
                "n_matches": int(sum(per_example)),
                "case_sensitive": self._case_sensitive,
                "strip_punctuation": self._strip_punctuation,
            },
        )

    def _is_match(self, generated: str, expected: str) -> bool:
        norm_gen = _normalise(
            generated,
            case_sensitive=self._case_sensitive,
            strip_punctuation=self._strip_punctuation,
        )
        norm_exp = _normalise(
            expected,
            case_sensitive=self._case_sensitive,
            strip_punctuation=self._strip_punctuation,
        )
        return norm_gen == norm_exp
