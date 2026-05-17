"""Token F1 metric: word-level overlap between generated and expected answers.

    Precision = |gen_tokens ∩ exp_tokens| / |gen_tokens|
    Recall    = |gen_tokens ∩ exp_tokens| / |exp_tokens|
    F1        = 2 * Precision * Recall / (Precision + Recall)

Tokens are multisets; tokenisation is a lowercase word-boundary regex.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult


def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _token_f1(generated: str, expected: str) -> Tuple[float, float, float]:
    """Return (precision, recall, f1) for one (generated, expected) pair."""
    gen_tokens = _tokenise(generated)
    exp_tokens = _tokenise(expected)

    if not gen_tokens and not exp_tokens:
        return 1.0, 1.0, 1.0
    if not gen_tokens or not exp_tokens:
        return 0.0, 0.0, 0.0

    gen_counts: Counter = Counter(gen_tokens)
    exp_counts: Counter = Counter(exp_tokens)

    overlap = sum((gen_counts & exp_counts).values())

    precision = overlap / sum(gen_counts.values())
    recall = overlap / sum(exp_counts.values())

    if precision + recall == 0.0:
        return precision, recall, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


class TokenF1Metric(BaseMetric):
    """Mean Token F1 across all predictions. Precision/recall reported in metadata."""

    @property
    def name(self) -> str:
        return "token_f1"

    def evaluate(self, predictions: List[EvaluationPrediction]) -> MetricResult:
        if not predictions:
            return MetricResult(
                metric_name=self.name,
                value=0.0,
                per_example=[],
                metadata={"n": 0, "note": "No predictions provided."},
            )

        per_f1: List[float] = []
        per_precision: List[float] = []
        per_recall: List[float] = []

        for pred in predictions:
            generated = (
                pred.generated_answer.text if pred.generated_answer is not None else ""
            )
            expected = pred.example.expected_answer
            p, r, f1 = _token_f1(generated, expected)
            per_f1.append(f1)
            per_precision.append(p)
            per_recall.append(r)

        mean_f1 = sum(per_f1) / len(per_f1)
        mean_precision = sum(per_precision) / len(per_precision)
        mean_recall = sum(per_recall) / len(per_recall)

        return MetricResult(
            metric_name=self.name,
            value=round(mean_f1, 6),
            per_example=[round(v, 6) for v in per_f1],
            metadata={
                "n": len(predictions),
                "mean_precision": round(mean_precision, 6),
                "mean_recall": round(mean_recall, 6),
            },
        )
