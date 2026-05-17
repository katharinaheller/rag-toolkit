"""Tests for the Exact Match metric."""

import pytest

from rag.evaluation.metrics.exact_match import ExactMatchMetric
from rag.evaluation.types import EvaluationExample, EvaluationPrediction, GeneratedAnswer, StageTiming


def _make_pred(query: str, expected: str, generated: str) -> EvaluationPrediction:
    example = EvaluationExample(
        query=query,
        expected_answer=expected,
        relevant_document_ids=["doc_1"],
    )
    answer = GeneratedAnswer(
        text=generated, model="test", strategy="test",
        latency_ms=0.0, prompt_chars=0, context_chars=0,
    )
    return EvaluationPrediction(
        example=example, retrieved_contexts=[], generated_answer=answer, timings=StageTiming(),
    )


class TestExactMatchMetric:
    def setup_method(self):
        self.metric = ExactMatchMetric()

    def test_perfect_match(self):
        pred = _make_pred("Q", "Paris", "paris")
        result = self.metric.evaluate([pred])
        assert result.value == 1.0
        assert result.per_example == [1.0]

    def test_no_match(self):
        pred = _make_pred("Q", "Paris", "London")
        result = self.metric.evaluate([pred])
        assert result.value == 0.0
        assert result.per_example == [0.0]

    def test_whitespace_normalisation(self):
        pred = _make_pred("Q", "hello world", "  hello   world  ")
        result = self.metric.evaluate([pred])
        assert result.value == 1.0

    def test_empty_predictions(self):
        result = self.metric.evaluate([])
        assert result.value == 0.0
        assert result.per_example == []
        assert result.metadata["n"] == 0

    def test_multiple_predictions(self):
        preds = [
            _make_pred("Q1", "Paris", "paris"),
            _make_pred("Q2", "Berlin", "London"),
            _make_pred("Q3", "Rome", "rome"),
        ]
        result = self.metric.evaluate(preds)
        assert result.value == pytest.approx(2 / 3)
        assert result.per_example == [1.0, 0.0, 1.0]

    def test_no_generated_answer(self):
        example = EvaluationExample(
            query="Q", expected_answer="Paris", relevant_document_ids=["doc_1"]
        )
        pred = EvaluationPrediction(
            example=example, retrieved_contexts=[], generated_answer=None, timings=StageTiming()
        )
        result = self.metric.evaluate([pred])
        assert result.value == 0.0

    def test_case_sensitive(self):
        metric = ExactMatchMetric(case_sensitive=True)
        pred = _make_pred("Q", "Paris", "paris")
        result = metric.evaluate([pred])
        assert result.value == 0.0

    def test_metadata_counts(self):
        preds = [
            _make_pred("Q1", "Paris", "paris"),
            _make_pred("Q2", "Berlin", "London"),
        ]
        result = self.metric.evaluate(preds)
        assert result.metadata["n"] == 2
        assert result.metadata["n_matches"] == 1
