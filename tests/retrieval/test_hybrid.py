"""Tests for HybridRetriever.

Hybrid retrieval delegates to a dense and a sparse leg, then applies fusion.
These tests focus on routing, leg independence, and the retrieve_with_details
contract used for diagnostic tracing.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from rag.retrieval.base import BaseRetriever
from rag.retrieval.config import FusionConfig
from rag.retrieval.hybrid import HybridRetriever


class FakeDenseLeg:
    """Stand-in for DenseRetriever."""

    def __init__(self, results, expected_k=None) -> None:
        self._results = results
        self.expected_k = expected_k
        self.calls: List[dict] = []

    def retrieve_candidates(self, query, k):
        self.calls.append({"query": query, "k": k})
        if self.expected_k is not None:
            assert k == self.expected_k
        return list(self._results)


class FakeSparseLeg(BaseRetriever):
    def __init__(self, results, expected_k=None) -> None:
        self._results = results
        self.expected_k = expected_k
        self.calls: List[dict] = []

    def retrieve_candidates(self, query, k):
        self.calls.append({"query": query, "k": k})
        if self.expected_k is not None:
            assert k == self.expected_k
        return list(self._results)


@pytest.fixture
def fusion_cfg() -> FusionConfig:
    return FusionConfig(strategy="weighted_sum", normalize_scores=False)


class TestHybridRouting:
    def test_non_dict_query_returns_empty(self, fusion_cfg: FusionConfig) -> None:
        dense = FakeDenseLeg([])
        sparse = FakeSparseLeg([])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        assert retriever.retrieve_candidates("not a dict", k=5) == []
        assert dense.calls == []
        assert sparse.calls == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_returns_empty(
        self, fusion_cfg: FusionConfig, k: int,
    ) -> None:
        dense = FakeDenseLeg([])
        sparse = FakeSparseLeg([])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        assert retriever.retrieve_candidates({"dense": [0.1]}, k=k) == []

    def test_only_dense_query_skips_sparse_leg(
        self, fusion_cfg: FusionConfig,
    ) -> None:
        dense = FakeDenseLeg(results=[{"id": "a", "score": 1.0}])
        sparse = FakeSparseLeg(results=[])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        retriever.retrieve_candidates({"dense": [0.1, 0.2]}, k=5)
        assert dense.calls and not sparse.calls

    def test_only_sparse_query_skips_dense_leg(
        self, fusion_cfg: FusionConfig,
    ) -> None:
        dense = FakeDenseLeg(results=[])
        sparse = FakeSparseLeg(results=[{"id": "b", "score": 1.0}])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        retriever.retrieve_candidates({"sparse": ["apple"]}, k=5)
        assert not dense.calls and sparse.calls

    def test_empty_sparse_value_skips_sparse_leg(
        self, fusion_cfg: FusionConfig,
    ) -> None:
        dense = FakeDenseLeg(results=[{"id": "a", "score": 1.0}])
        sparse = FakeSparseLeg(results=[])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        retriever.retrieve_candidates({"dense": [0.1], "sparse": []}, k=5)
        assert not sparse.calls

    def test_dense_empty_vector_skips_dense_leg(
        self, fusion_cfg: FusionConfig,
    ) -> None:
        dense = FakeDenseLeg(results=[])
        sparse = FakeSparseLeg(results=[{"id": "b", "score": 1.0}])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        retriever.retrieve_candidates({"dense": [], "sparse": ["x"]}, k=5)
        assert not dense.calls


class TestHybridCandidateK:
    def test_uses_per_leg_candidate_k(self, fusion_cfg: FusionConfig) -> None:
        dense = FakeDenseLeg(results=[], expected_k=37)
        sparse = FakeSparseLeg(results=[], expected_k=42)
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 37, 42)
        retriever.retrieve_candidates({"dense": [0.1], "sparse": ["a"]}, k=5)


class TestRetrieveWithDetails:
    def test_returns_all_three_lists(self, fusion_cfg: FusionConfig) -> None:
        dense = FakeDenseLeg(results=[
            {"id": "a", "score": 0.9},
            {"id": "b", "score": 0.5},
        ])
        sparse = FakeSparseLeg(results=[
            {"id": "b", "score": 0.7},
            {"id": "c", "score": 0.6},
        ])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        d, s, f = retriever.retrieve_with_details(
            {"dense": [0.1], "sparse": ["t"]}, k=10
        )
        assert [r["id"] for r in d] == ["a", "b"]
        assert [r["id"] for r in s] == ["b", "c"]
        assert {r["id"] for r in f} == {"a", "b", "c"}

    def test_details_for_invalid_query(self, fusion_cfg: FusionConfig) -> None:
        dense = FakeDenseLeg(results=[])
        sparse = FakeSparseLeg(results=[])
        retriever = HybridRetriever(dense, sparse, fusion_cfg, 50, 50)
        d, s, f = retriever.retrieve_with_details("nope", k=10)
        assert d == [] and s == [] and f == []
