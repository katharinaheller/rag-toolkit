"""Tests for the Throughput metric."""

import pytest

from rag.evaluation.metrics.throughput import ThroughputMetric
from rag.evaluation.types import EvaluationExample, EvaluationPrediction, StageTiming


def _pred():
    ex = EvaluationExample(query="Q", expected_answer="A", relevant_document_ids=[])
    return EvaluationPrediction(
        example=ex, retrieved_contexts=[], generated_answer=None, timings=StageTiming()
    )


class TestThroughputMetric:
    def test_basic_throughput(self):
        metric = ThroughputMetric(total_duration_s=10.0)
        preds = [_pred() for _ in range(100)]
        result = metric.evaluate(preds)
        assert result.value == pytest.approx(10.0)

    def test_zero_duration(self):
        metric = ThroughputMetric(total_duration_s=0.0)
        preds = [_pred()]
        result = metric.evaluate(preds)
        assert result.value == 0.0
        assert "note" in result.metadata

    def test_none_duration(self):
        metric = ThroughputMetric(total_duration_s=None)
        result = metric.evaluate([_pred()])
        assert result.value == 0.0

    def test_empty_predictions(self):
        metric = ThroughputMetric(total_duration_s=5.0)
        result = metric.evaluate([])
        assert result.value == 0.0

    def test_unit_in_metadata(self):
        metric = ThroughputMetric(total_duration_s=1.0)
        result = metric.evaluate([_pred()])
        assert result.metadata["unit"] == "examples_per_second"

    def test_fractional_throughput(self):
        metric = ThroughputMetric(total_duration_s=3.0)
        preds = [_pred() for _ in range(1)]
        result = metric.evaluate(preds)
        assert result.value == pytest.approx(1 / 3, rel=1e-3)
