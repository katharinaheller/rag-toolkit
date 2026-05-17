"""Tests for the Context Recall metric."""

import pytest

from rag.evaluation.metrics.context_recall import ContextRecallMetric
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, RetrievedContext, StageTiming,
)


def _ctx(doc_id, chunk_id, rank=0):
    return RetrievedContext(document_id=doc_id, chunk_id=chunk_id, text="t", score=1.0, rank=rank)


def _pred(relevant_doc_ids, retrieved_doc_ids, relevant_chunk_ids=None):
    ex = EvaluationExample(
        query="Q", expected_answer="A",
        relevant_document_ids=relevant_doc_ids,
        relevant_chunk_ids=relevant_chunk_ids,
    )
    ctxs = [_ctx(did, f"c_{did}", i) for i, did in enumerate(retrieved_doc_ids)]
    return EvaluationPrediction(
        example=ex, retrieved_contexts=ctxs,
        generated_answer=None, timings=StageTiming()
    )


class TestContextRecall:
    def setup_method(self):
        self.metric = ContextRecallMetric()

    def test_full_recall(self):
        result = self.metric.evaluate([_pred(["doc_1", "doc_2"], ["doc_1", "doc_2"])])
        assert result.value == pytest.approx(1.0)

    def test_zero_recall(self):
        result = self.metric.evaluate([_pred(["doc_1"], ["doc_2", "doc_3"])])
        assert result.value == pytest.approx(0.0)

    def test_partial_recall(self):
        result = self.metric.evaluate([_pred(["doc_1", "doc_2"], ["doc_1"])])
        assert result.value == pytest.approx(0.5)

    def test_empty_relevant(self):
        """No relevant labels → recall = 0.0 with a warning in metadata."""
        result = self.metric.evaluate([_pred([], ["doc_1"])])
        assert result.value == pytest.approx(0.0)
        assert "warning" in result.metadata

    def test_empty_retrieved(self):
        result = self.metric.evaluate([_pred(["doc_1"], [])])
        assert result.value == pytest.approx(0.0)

    def test_empty_predictions(self):
        result = self.metric.evaluate([])
        assert result.value == 0.0
        assert result.per_example == []

    def test_chunk_level_recall(self):
        ex = EvaluationExample(
            query="Q", expected_answer="A",
            relevant_document_ids=["doc_1"],
            relevant_chunk_ids=["c_1", "c_2"],
        )
        ctxs = [
            _ctx("doc_1", "c_1", 0),
            _ctx("doc_1", "c_3", 1),
        ]
        pred = EvaluationPrediction(
            example=ex, retrieved_contexts=ctxs,
            generated_answer=None, timings=StageTiming()
        )
        result = self.metric.evaluate([pred])
        # 1 of 2 relevant chunks retrieved
        assert result.value == pytest.approx(0.5)
