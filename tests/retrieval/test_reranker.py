"""Tests for BaselineReranker.

The reranker is a pure transformation over RetrievalResult lists. Tests check
the disabled/enabled paths, score blending, sort order, and metadata
preservation (dense_score, sparse_score must survive the rerank step).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from rag.retrieval.config import RerankerConfig
from rag.retrieval.reranker import BaselineReranker


def _result(rid: str, text: str, score: float, **extra) -> Dict[str, Any]:
    return {
        "id": rid,
        "chunk_id": f"c_{rid}",
        "document_id": f"doc_{rid}",
        "score": score,
        "retrieval_score": score,
        "text": text,
        "metadata": {"source": "synthetic"},
        **extra,
    }


class TestDisabled:
    def test_returns_input_unchanged(self) -> None:
        cfg = RerankerConfig(enabled=False)
        rr = BaselineReranker(cfg)
        results = [_result("a", "alpha", 1.0)]
        out = rr.rerank("alpha", results)
        assert out is results

    def test_empty_results_passthrough(self) -> None:
        rr = BaselineReranker(RerankerConfig(enabled=True))
        assert rr.rerank("query", []) == []


class TestEnabled:
    def test_lexical_overlap_boosts_matching_doc(self) -> None:
        cfg = RerankerConfig(
            enabled=True, lexical_weight=1.0, length_weight=0.0, target_length=10,
        )
        rr = BaselineReranker(cfg)
        results = [
            _result("a", "the quick brown fox", 0.0),
            _result("b", "nothing matches", 0.0),
        ]
        out = rr.rerank("quick brown", results)
        out_map = {r["id"]: r for r in out}
        assert out_map["a"]["score"] > out_map["b"]["score"]

    def test_length_score_peaks_at_target(self) -> None:
        cfg = RerankerConfig(
            enabled=True, lexical_weight=0.0, length_weight=1.0, target_length=20,
        )
        rr = BaselineReranker(cfg)
        results = [
            _result("at_target", "x" * 20, 0.0),
            _result("too_short", "xx", 0.0),
            _result("too_long", "x" * 200, 0.0),
        ]
        out = rr.rerank("q", results)
        scores = {r["id"]: r["score"] for r in out}
        assert scores["at_target"] > scores["too_short"]
        assert scores["at_target"] > scores["too_long"]

    def test_rerank_delta_recorded(self) -> None:
        cfg = RerankerConfig(enabled=True, lexical_weight=0.5, length_weight=0.5)
        rr = BaselineReranker(cfg)
        results = [_result("a", "alpha beta gamma", 0.1)]
        out = rr.rerank("alpha", results)
        assert "rerank_score" in out[0]
        assert out[0]["retrieval_score"] == 0.1
        assert out[0]["score"] == pytest.approx(
            out[0]["retrieval_score"] + out[0]["rerank_score"], rel=1e-9,
        )

    def test_preserves_dense_and_sparse_scores(self) -> None:
        cfg = RerankerConfig(enabled=True)
        rr = BaselineReranker(cfg)
        results = [
            _result("a", "alpha", 0.5, dense_score=0.4, sparse_score=0.6),
        ]
        out = rr.rerank("alpha", results)
        assert out[0]["dense_score"] == 0.4
        assert out[0]["sparse_score"] == 0.6

    def test_results_sorted_after_rerank(self) -> None:
        cfg = RerankerConfig(enabled=True, lexical_weight=1.0, length_weight=0.0)
        rr = BaselineReranker(cfg)
        # Same retrieval_score; only lexical overlap differentiates.
        results = [
            _result("b", "no match here", 1.0),
            _result("a", "alpha", 1.0),
        ]
        out = rr.rerank("alpha", results)
        assert out[0]["id"] == "a"

    def test_empty_query_yields_zero_overlap(self) -> None:
        cfg = RerankerConfig(enabled=True, lexical_weight=1.0, length_weight=0.0)
        rr = BaselineReranker(cfg)
        results = [_result("a", "alpha", 0.5)]
        out = rr.rerank("", results)
        # No query tokens → overlap contributes 0.
        assert out[0]["rerank_score"] == pytest.approx(0.0)


class TestDeterminism:
    def test_repeated_calls_yield_identical_output(self) -> None:
        cfg = RerankerConfig(enabled=True)
        rr = BaselineReranker(cfg)
        results = [_result("a", "alpha beta", 1.0), _result("b", "beta", 0.5)]
        a = rr.rerank("alpha", list(results))
        b = rr.rerank("alpha", list(results))
        assert a == b
