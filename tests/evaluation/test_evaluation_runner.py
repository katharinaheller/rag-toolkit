"""Integration tests for the EvaluationRunner.

Uses mock retriever and strategy to avoid any external dependency.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from rag.evaluation.config import EvaluationConfig
from rag.evaluation.runner import EvaluationRunner
from rag.evaluation.types import EvaluationExample


# ── Minimal mocks ────────────────────────────────────────────────────────────

class _FakeRetriever:
    """Returns a fixed list of retrieval results for any query."""

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        return [
            {
                "id": "emb_1",
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "score": 0.9,
                "retrieval_score": 0.9,
                "text": "RAG stands for Retrieval-Augmented Generation.",
                "metadata": {},
            }
        ]


class _FakeStrategy:
    """Returns a fixed generation result for any input."""

    def generate(self, query: str, context_chunks: List[str]):
        return _FakeGenerationResult(
            answer="Retrieval-Augmented Generation",
            model="fake_model",
            strategy="FakeStrategy",
            latency_ms=10.0,
            prompt_chars=100,
            context_chars=50,
            error=None,
            raw_response=None,
        )


class _FakeGenerationResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FailingRetriever:
    """Always raises during retrieve()."""
    def retrieve(self, query: str, k: int = 5):
        raise RuntimeError("Simulated retrieval failure")


class _FailingStrategy:
    """Always raises during generate()."""
    def generate(self, query, context_chunks):
        raise RuntimeError("Simulated generation failure")


def _examples():
    return [
        EvaluationExample(
            query="What is RAG?",
            expected_answer="Retrieval-Augmented Generation",
            relevant_document_ids=["doc_1"],
        ),
        EvaluationExample(
            query="What is BGE-M3?",
            expected_answer="A multi-function embedding model.",
            relevant_document_ids=["doc_2"],
        ),
    ]


# ── Tests ────────────────────────────────────────────────────────────────────

class TestEvaluationRunner:
    def test_retrieval_only(self):
        config = EvaluationConfig(mode="retrieval", top_k=5)
        runner = EvaluationRunner(config, retriever=_FakeRetriever())
        result = runner.run(_examples())
        assert result.n_examples == 2
        for pred in result.predictions:
            assert len(pred.retrieved_contexts) == 1
            assert pred.generated_answer is None

    def test_end_to_end(self):
        config = EvaluationConfig(mode="end_to_end", top_k=5)
        runner = EvaluationRunner(
            config, retriever=_FakeRetriever(), strategy=_FakeStrategy()
        )
        result = runner.run(_examples())
        assert result.n_examples == 2
        for pred in result.predictions:
            assert pred.generated_answer is not None
            assert pred.generated_answer.text == "Retrieval-Augmented Generation"

    def test_generation_only(self):
        config = EvaluationConfig(mode="generation", top_k=5)
        runner = EvaluationRunner(config, strategy=_FakeStrategy())
        result = runner.run(_examples())
        assert result.n_examples == 2

    def test_metrics_computed(self):
        config = EvaluationConfig(
            mode="end_to_end",
            retrieval_metrics=["context_precision", "context_recall", "mrr"],
            generation_metrics=["exact_match"],
            performance_metrics=["latency"],
        )
        runner = EvaluationRunner(
            config, retriever=_FakeRetriever(), strategy=_FakeStrategy()
        )
        result = runner.run(_examples())
        assert "exact_match" in result.metric_results
        assert "context_precision" in result.metric_results
        assert "mrr" in result.metric_results
        assert "latency" in result.metric_results

    def test_retrieval_error_captured(self):
        """Retrieval errors are captured, not propagated."""
        config = EvaluationConfig(mode="retrieval")
        runner = EvaluationRunner(config, retriever=_FailingRetriever())
        result = runner.run([_examples()[0]])
        assert result.n_examples == 1
        pred = result.predictions[0]
        assert len(pred.errors) == 1
        assert "retrieval error" in pred.errors[0]

    def test_generation_error_captured(self):
        """Generation errors are captured, not propagated."""
        config = EvaluationConfig(mode="end_to_end")
        runner = EvaluationRunner(
            config, retriever=_FakeRetriever(), strategy=_FailingStrategy()
        )
        result = runner.run([_examples()[0]])
        pred = result.predictions[0]
        assert any("generation error" in e for e in pred.errors)

    def test_requires_retriever_for_retrieval_mode(self):
        with pytest.raises(ValueError, match="retriever"):
            EvaluationRunner(EvaluationConfig(mode="retrieval"), strategy=_FakeStrategy())

    def test_requires_strategy_for_generation_mode(self):
        with pytest.raises(ValueError, match="strategy"):
            EvaluationRunner(EvaluationConfig(mode="generation"), retriever=_FakeRetriever())

    def test_run_id_in_result(self):
        config = EvaluationConfig(mode="retrieval", run_id="my_custom_run")
        runner = EvaluationRunner(config, retriever=_FakeRetriever())
        result = runner.run(_examples())
        assert result.run_id == "my_custom_run"

    def test_duration_is_positive(self):
        config = EvaluationConfig(mode="retrieval")
        runner = EvaluationRunner(config, retriever=_FakeRetriever())
        result = runner.run(_examples())
        assert result.duration_s >= 0.0

    def test_empty_dataset(self):
        config = EvaluationConfig(mode="retrieval")
        runner = EvaluationRunner(config, retriever=_FakeRetriever())
        result = runner.run([])
        assert result.n_examples == 0
