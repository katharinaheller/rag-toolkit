"""Tests for the Latency metric."""

import pytest

from rag.evaluation.metrics.latency import LatencyMetric
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, GeneratedAnswer, StageTiming,
)


def _pred(end_to_end_ms=None, retrieval_ms=None, generation_ms=None):
    ex = EvaluationExample(query="Q", expected_answer="A", relevant_document_ids=[])
    answer = GeneratedAnswer(
        text="A", model="m", strategy="s",
        latency_ms=generation_ms or 0.0,
        prompt_chars=10, context_chars=5,
    ) if generation_ms is not None else None
    timings = StageTiming(
        retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
        end_to_end_ms=end_to_end_ms,
    )
    return EvaluationPrediction(
        example=ex, retrieved_contexts=[], generated_answer=answer, timings=timings
    )


class TestLatencyMetric:
    def setup_method(self):
        self.metric = LatencyMetric()

    def test_empty_predictions(self):
        result = self.metric.evaluate([])
        assert result.value == 0.0

    def test_mean_end_to_end(self):
        preds = [_pred(end_to_end_ms=100.0), _pred(end_to_end_ms=200.0)]
        result = self.metric.evaluate(preds)
        assert result.value == pytest.approx(150.0)

    def test_per_stage_metadata(self):
        pred = _pred(end_to_end_ms=500.0, retrieval_ms=100.0, generation_ms=300.0)
        result = self.metric.evaluate([pred])
        assert result.metadata["mean_retrieval_ms"] == pytest.approx(100.0)
        assert result.metadata["mean_generation_ms"] == pytest.approx(300.0)
        assert result.metadata["mean_end_to_end_ms"] == pytest.approx(500.0)

    def test_none_timings_excluded(self):
        """Predictions with None timings should not crash the metric."""
        pred = _pred()  # All timings None
        result = self.metric.evaluate([pred])
        # No end_to_end_ms, no generation_ms → value = 0.0
        assert result.value == pytest.approx(0.0)

    def test_unit_in_metadata(self):
        result = self.metric.evaluate([_pred(end_to_end_ms=10.0)])
        assert result.metadata["unit"] == "milliseconds"
