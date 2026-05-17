"""Tests for DenseRetriever.

DenseRetriever forwards a dense vector query to a BaseDenseIndex and applies
the canonical (-score, id) ordering. The tests pin the contract by exercising
edge cases for k, query type validation, candidate_k expansion and ordering.
"""

from __future__ import annotations

from typing import List

import pytest

from rag.retrieval.config import DenseRetrievalConfig
from rag.retrieval.dense import DenseRetriever


class FakeDenseIndex:
    """In-memory BaseDenseIndex stand-in returning a configured candidate list."""

    def __init__(self, results, expected_k=None) -> None:
        self._results = results
        self.expected_k = expected_k
        self.queries: List[dict] = []

    def add(self, entries) -> None:
        raise NotImplementedError

    def query(self, query, k):
        self.queries.append({"query": list(query), "k": k})
        if self.expected_k is not None:
            assert k == self.expected_k, f"expected k={self.expected_k}, got {k}"
        return list(self._results)

    def persist(self) -> None:
        pass

    def load(self) -> None:
        pass


@pytest.fixture
def cfg() -> DenseRetrievalConfig:
    return DenseRetrievalConfig(candidate_k=100)


class TestQueryShape:
    def test_empty_query_returns_empty(self, cfg: DenseRetrievalConfig) -> None:
        idx = FakeDenseIndex(results=[])
        retriever = DenseRetriever(idx, cfg)
        assert retriever.retrieve_candidates([], k=5) == []
        assert idx.queries == []

    def test_non_list_query_returns_empty(self, cfg: DenseRetrievalConfig) -> None:
        idx = FakeDenseIndex(results=[])
        retriever = DenseRetriever(idx, cfg)
        assert retriever.retrieve_candidates("not-a-list", k=5) == []
        assert retriever.retrieve_candidates({"dense": [0.1]}, k=5) == []
        assert idx.queries == []

    @pytest.mark.parametrize("k", [0, -1, -100])
    def test_non_positive_k_returns_empty(self, cfg: DenseRetrievalConfig, k: int) -> None:
        idx = FakeDenseIndex(results=[{"id": "a", "score": 1.0}])
        retriever = DenseRetriever(idx, cfg)
        assert retriever.retrieve_candidates([0.1, 0.2], k=k) == []


class TestCandidateExpansion:
    def test_fetches_at_least_candidate_k(self) -> None:
        cfg = DenseRetrievalConfig(candidate_k=50)
        idx = FakeDenseIndex(results=[], expected_k=50)
        retriever = DenseRetriever(idx, cfg)
        retriever.retrieve_candidates([0.1], k=5)

    def test_fetches_at_least_k_when_larger_than_candidate_k(self) -> None:
        cfg = DenseRetrievalConfig(candidate_k=10)
        idx = FakeDenseIndex(results=[], expected_k=100)
        retriever = DenseRetriever(idx, cfg)
        retriever.retrieve_candidates([0.1], k=100)


class TestOrderingAndTruncation:
    def test_results_sorted_by_score_desc_then_id_asc(self, cfg: DenseRetrievalConfig) -> None:
        idx = FakeDenseIndex(results=[
            {"id": "b", "score": 0.5},
            {"id": "a", "score": 0.9},
            {"id": "c", "score": 0.9},
        ])
        retriever = DenseRetriever(idx, cfg)
        out = retriever.retrieve_candidates([0.1], k=3)
        # tie on 0.9 → id asc; 0.5 last
        assert [r["id"] for r in out] == ["a", "c", "b"]

    def test_truncates_to_k(self, cfg: DenseRetrievalConfig) -> None:
        idx = FakeDenseIndex(results=[
            {"id": f"d{i}", "score": float(10 - i)} for i in range(10)
        ])
        retriever = DenseRetriever(idx, cfg)
        out = retriever.retrieve_candidates([0.1], k=3)
        assert len(out) == 3
        assert out[0]["id"] == "d0"

    def test_strips_extra_fields(self, cfg: DenseRetrievalConfig) -> None:
        idx = FakeDenseIndex(results=[
            {"id": "a", "score": 1.0, "extra": "ignored"}
        ])
        retriever = DenseRetriever(idx, cfg)
        out = retriever.retrieve_candidates([0.1], k=1)
        assert out[0] == {"id": "a", "score": 1.0}
