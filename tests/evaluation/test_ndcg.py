"""Tests for the nDCG metric."""

import math
import pytest

from rag.evaluation.metrics.ndcg import NDCGMetric, _dcg, _ndcg_for_prediction
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


class TestNDCG:
    def test_ideal_ranking(self):
        """Relevant items at top → nDCG = 1.0."""
        metric = NDCGMetric()
        result = metric.evaluate([_pred(["doc_1", "doc_2"], ["doc_1", "doc_2", "doc_3"])])
        assert result.value == pytest.approx(1.0, abs=1e-4)

    def test_reversed_ranking(self):
        """Relevant items at bottom → nDCG < 1.0."""
        metric = NDCGMetric()
        result = metric.evaluate([_pred(["doc_3"], ["doc_1", "doc_2", "doc_3"])])
        assert 0.0 < result.value < 1.0

    def test_no_relevant_retrieved(self):
        metric = NDCGMetric()
        result = metric.evaluate([_pred(["doc_x"], ["doc_1", "doc_2"])])
        assert result.value == pytest.approx(0.0)

    def test_empty_relevant(self):
        metric = NDCGMetric()
        result = metric.evaluate([_pred([], ["doc_1"])])
        assert result.value == pytest.approx(0.0)

    def test_empty_predictions(self):
        metric = NDCGMetric()
        result = metric.evaluate([])
        assert result.value == 0.0
        assert result.per_example == []

    def test_k_cutoff(self):
        """nDCG@1 should only consider the top result."""
        metric = NDCGMetric(k=1)
        result = metric.evaluate([_pred(["doc_2"], ["doc_1", "doc_2"])])
        # doc_2 is not at rank 0 → nDCG@1 = 0.0
        assert result.value == pytest.approx(0.0)

    def test_k_cutoff_hit(self):
        metric = NDCGMetric(k=1)
        result = metric.evaluate([_pred(["doc_1"], ["doc_1", "doc_2"])])
        assert result.value == pytest.approx(1.0)

    def test_metric_name_includes_k(self):
        metric = NDCGMetric(k=10)
        assert "@10" in metric.name

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            NDCGMetric(k=0)

    def test_dcg_helper(self):
        # DCG([1, 0, 1]) = 1/log2(2) + 0/log2(3) + 1/log2(4)
        expected = 1.0 / math.log2(2) + 1.0 / math.log2(4)
        assert _dcg([1, 0, 1]) == pytest.approx(expected)
