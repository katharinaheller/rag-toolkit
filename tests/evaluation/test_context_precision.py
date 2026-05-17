"""Tests for the Context Precision metric."""

import pytest

from rag.evaluation.metrics.context_precision import ContextPrecisionMetric
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, RetrievedContext, StageTiming,
)


def _make_example(relevant_doc_ids, relevant_chunk_ids=None):
    return EvaluationExample(
        query="Q",
        expected_answer="A",
        relevant_document_ids=relevant_doc_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )


def _make_ctx(doc_id: str, chunk_id: str, rank: int) -> RetrievedContext:
    return RetrievedContext(
        document_id=doc_id, chunk_id=chunk_id, text="t", score=1.0, rank=rank
    )


def _make_pred(example, contexts):
    return EvaluationPrediction(
        example=example, retrieved_contexts=contexts,
        generated_answer=None, timings=StageTiming()
    )


class TestContextPrecision:
    def setup_method(self):
        self.metric = ContextPrecisionMetric()

    def test_all_relevant(self):
        ex = _make_example(["doc_1", "doc_2"])
        ctxs = [_make_ctx("doc_1", "c1", 0), _make_ctx("doc_2", "c2", 1)]
        result = self.metric.evaluate([_make_pred(ex, ctxs)])
        assert result.value == pytest.approx(1.0)

    def test_none_relevant(self):
        ex = _make_example(["doc_x"])
        ctxs = [_make_ctx("doc_1", "c1", 0), _make_ctx("doc_2", "c2", 1)]
        result = self.metric.evaluate([_make_pred(ex, ctxs)])
        assert result.value == pytest.approx(0.0)

    def test_partial_precision(self):
        ex = _make_example(["doc_1"])
        ctxs = [_make_ctx("doc_1", "c1", 0), _make_ctx("doc_2", "c2", 1)]
        result = self.metric.evaluate([_make_pred(ex, ctxs)])
        assert result.value == pytest.approx(0.5)

    def test_empty_retrieved(self):
        ex = _make_example(["doc_1"])
        result = self.metric.evaluate([_make_pred(ex, [])])
        assert result.value == pytest.approx(0.0)

    def test_empty_predictions(self):
        result = self.metric.evaluate([])
        assert result.value == 0.0
        assert result.per_example == []

    def test_duplicate_document_ids(self):
        """Duplicate document IDs should be deduplicated before scoring."""
        ex = _make_example(["doc_1"])
        ctxs = [
            _make_ctx("doc_1", "c1", 0),
            _make_ctx("doc_1", "c2", 1),  # Duplicate document_id
        ]
        result = self.metric.evaluate([_make_pred(ex, ctxs)])
        # Only 1 unique doc, 1 relevant → precision = 1.0
        assert result.value == pytest.approx(1.0)

    def test_chunk_level_matching(self):
        ex = _make_example(
            relevant_doc_ids=["doc_1"],
            relevant_chunk_ids=["c1"]
        )
        ctxs = [_make_ctx("doc_1", "c1", 0), _make_ctx("doc_1", "c2", 1)]
        result = self.metric.evaluate([_make_pred(ex, ctxs)])
        # 1 relevant chunk out of 2 unique chunks
        assert result.value == pytest.approx(0.5)
