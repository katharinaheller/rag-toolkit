"""Tests for BM25Retriever.

These tests integrate against the real SparseIndex and InvertedIndexView to
avoid mocking the very thing being scored. Synthetic corpora are constructed
in-memory; persistence is not required for scoring correctness.
"""

from __future__ import annotations

from typing import List

import pytest

from rag.indexing.config import SparseIndexConfig
from rag.indexing.sparse.sparse_index import SparseIndex
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.retrieval.bm25 import BM25Retriever
from rag.retrieval.config import BM25Config


@pytest.fixture
def sparse_components(tmp_path):
    """Build a SparseIndex with a small but discriminating corpus."""
    cfg = SparseIndexConfig(tokenizer="simple")
    idx = SparseIndex(cfg, tmp_path / "sparse")
    tokenizer = Tokenizer(cfg)
    docs = [
        ("d_apple", "apple banana cherry"),
        ("d_apple2", "apple apple apple banana"),
        ("d_banana", "banana banana fig"),
        ("d_cherry", "cherry cherry fig date"),
        ("d_date", "date elderberry fig grape"),
    ]
    entries = [{"id": did, "tokens": tokenizer.tokenize(text)} for did, text in docs]
    idx.add(entries)
    view = idx.get_view()
    return idx, view, tokenizer


class TestBM25Inputs:
    def test_string_query_is_tokenized(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        out = retriever.retrieve_candidates("apple", k=10)
        ids = [r["id"] for r in out]
        assert "d_apple" in ids
        assert "d_apple2" in ids

    def test_list_query_is_used_directly(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        out = retriever.retrieve_candidates(["apple"], k=10)
        ids = [r["id"] for r in out]
        assert "d_apple" in ids

    def test_invalid_query_type_returns_empty(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        assert retriever.retrieve_candidates(123, k=5) == []
        assert retriever.retrieve_candidates(None, k=5) == []

    def test_empty_query_returns_empty(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        assert retriever.retrieve_candidates("", k=5) == []
        assert retriever.retrieve_candidates([], k=5) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_returns_empty(self, sparse_components, k: int) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        assert retriever.retrieve_candidates("apple", k=k) == []


class TestBM25Scoring:
    def test_more_frequent_term_scores_higher(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        out = retriever.retrieve_candidates("apple", k=10)
        scored = {r["id"]: r["score"] for r in out}
        # d_apple2 has three occurrences; d_apple has one — TF saturation still
        # leaves d_apple2 ahead given equal length normalisation.
        assert scored["d_apple2"] > scored["d_apple"]

    def test_results_sorted_by_score_desc(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        out = retriever.retrieve_candidates("banana fig", k=10)
        scores = [r["score"] for r in out]
        assert scores == sorted(scores, reverse=True)

    def test_duplicate_query_tokens_count_once(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        once = retriever.retrieve_candidates("apple", k=10)
        twice = retriever.retrieve_candidates("apple apple apple", k=10)
        once_map = {r["id"]: r["score"] for r in once}
        twice_map = {r["id"]: r["score"] for r in twice}
        for did in once_map:
            assert once_map[did] == pytest.approx(twice_map[did])

    def test_unknown_token_yields_no_matches(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        assert retriever.retrieve_candidates("zzz_unknown", k=10) == []

    def test_results_truncated_to_k(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = BM25Retriever(idx, view, tok, BM25Config())
        out = retriever.retrieve_candidates("fig", k=2)
        assert len(out) <= 2
