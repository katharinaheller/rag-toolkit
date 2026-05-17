"""Tests for the Token F1 metric."""

import pytest

from rag.evaluation.metrics.token_f1 import TokenF1Metric, _token_f1
from rag.evaluation.types import EvaluationExample, EvaluationPrediction, GeneratedAnswer, StageTiming


def _make_pred(expected: str, generated: str) -> EvaluationPrediction:
    example = EvaluationExample(
        query="Q", expected_answer=expected, relevant_document_ids=["doc_1"]
    )
    answer = GeneratedAnswer(
        text=generated, model="test", strategy="test",
        latency_ms=0.0, prompt_chars=0, context_chars=0,
    )
    return EvaluationPrediction(
        example=example, retrieved_contexts=[], generated_answer=answer, timings=StageTiming()
    )


class TestTokenF1:
    def test_exact_match(self):
        p, r, f1 = _token_f1("hello world", "hello world")
        assert f1 == pytest.approx(1.0)

    def test_no_overlap(self):
        p, r, f1 = _token_f1("foo bar", "baz qux")
        assert f1 == pytest.approx(0.0)

    def test_partial_overlap(self):
        p, r, f1 = _token_f1("the cat sat", "the cat")
        assert f1 == pytest.approx(2 * (2/3) * (2/2) / ((2/3) + (2/2)))

    def test_both_empty(self):
        p, r, f1 = _token_f1("", "")
        assert f1 == pytest.approx(1.0)

    def test_one_empty(self):
        p, r, f1 = _token_f1("hello", "")
        assert f1 == pytest.approx(0.0)

    def test_empty_predictions_list(self):
        metric = TokenF1Metric()
        result = metric.evaluate([])
        assert result.value == 0.0
        assert result.per_example == []

    def test_aggregate(self):
        metric = TokenF1Metric()
        preds = [
            _make_pred("hello world", "hello world"),  # F1 = 1.0
            _make_pred("foo bar", "baz"),               # F1 = 0.0
        ]
        result = metric.evaluate(preds)
        assert result.value == pytest.approx(0.5)

    def test_metadata_includes_precision_recall(self):
        metric = TokenF1Metric()
        pred = _make_pred("hello world", "hello world")
        result = metric.evaluate([pred])
        assert "mean_precision" in result.metadata
        assert "mean_recall" in result.metadata

    def test_no_generated_answer_counts_zero(self):
        metric = TokenF1Metric()
        example = EvaluationExample(
            query="Q", expected_answer="answer", relevant_document_ids=[]
        )
        pred = EvaluationPrediction(
            example=example, retrieved_contexts=[], generated_answer=None, timings=StageTiming()
        )
        result = metric.evaluate([pred])
        assert result.value == pytest.approx(0.0)
