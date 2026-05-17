"""Tests for the MRR metric."""

import pytest

from rag.evaluation.metrics.mrr import MRRMetric
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, RetrievedContext, StageTiming,
)


def _pred(relevant_doc_ids, retrieved_doc_ids):
    ex = EvaluationExample(
        query="Q", expected_answer="A", relevant_document_ids=relevant_doc_ids
    )
    ctxs = [
        RetrievedContext(document_id=did, chunk_id=f"c{i}", text="t", score=1.0, rank=i)
        for i, did in enumerate(retrieved_doc_ids)
    ]
    return EvaluationPrediction(
        example=ex, retrieved_contexts=ctxs,
        generated_answer=None, timings=StageTiming()
    )


class TestMRR:
    def setup_method(self):
        self.metric = MRRMetric()

    def test_first_result_relevant(self):
        result = self.metric.evaluate([_pred(["doc_1"], ["doc_1", "doc_2"])])
        assert result.value == pytest.approx(1.0)

    def test_second_result_relevant(self):
        result = self.metric.evaluate([_pred(["doc_2"], ["doc_1", "doc_2"])])
        assert result.value == pytest.approx(0.5)

    def test_third_result_relevant(self):
        result = self.metric.evaluate([_pred(["doc_3"], ["doc_1", "doc_2", "doc_3"])])
        assert result.value == pytest.approx(1 / 3)

    def test_no_relevant_retrieved(self):
        result = self.metric.evaluate([_pred(["doc_x"], ["doc_1", "doc_2"])])
        assert result.value == pytest.approx(0.0)

    def test_empty_retrieved(self):
        result = self.metric.evaluate([_pred(["doc_1"], [])])
        assert result.value == pytest.approx(0.0)

    def test_empty_predictions(self):
        result = self.metric.evaluate([])
        assert result.value == 0.0
        assert result.per_example == []

    def test_mean_over_multiple(self):
        preds = [
            _pred(["doc_1"], ["doc_1"]),         # RR = 1.0
            _pred(["doc_2"], ["doc_1", "doc_2"]), # RR = 0.5
        ]
        result = self.metric.evaluate(preds)
        assert result.value == pytest.approx(0.75)

    def test_no_relevant_labels(self):
        result = self.metric.evaluate([_pred([], ["doc_1"])])
        assert result.value == pytest.approx(0.0)

    def test_duplicate_doc_ids_skipped(self):
        """A duplicate document ID should not count as the first relevant hit."""
        ex = EvaluationExample(
            query="Q", expected_answer="A", relevant_document_ids=["doc_1"]
        )
        ctxs = [
            RetrievedContext(document_id="doc_2", chunk_id="c0", text="t", score=1.0, rank=0),
            RetrievedContext(document_id="doc_2", chunk_id="c1", text="t", score=0.9, rank=1),
            RetrievedContext(document_id="doc_1", chunk_id="c2", text="t", score=0.8, rank=2),
        ]
        pred = EvaluationPrediction(
            example=ex, retrieved_contexts=ctxs,
            generated_answer=None, timings=StageTiming()
        )
        result = self.metric.evaluate([pred])
        # doc_2 appears twice (deduplicated), doc_1 is at position 2 → RR = 1/2
        assert result.value == pytest.approx(0.5)
