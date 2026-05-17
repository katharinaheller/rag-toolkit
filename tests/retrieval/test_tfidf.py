"""Tests for TFIDFRetriever.

Verifies sublinear vs linear TF, IDF behaviour, and the deterministic
(-score, id) ordering on a small in-memory corpus.
"""

from __future__ import annotations

import pytest

from rag.indexing.config import SparseIndexConfig
from rag.indexing.sparse.sparse_index import SparseIndex
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.retrieval.config import TFIDFConfig
from rag.retrieval.tfidf import TFIDFRetriever


@pytest.fixture
def sparse_components(tmp_path):
    cfg = SparseIndexConfig(tokenizer="simple")
    idx = SparseIndex(cfg, tmp_path / "sparse")
    tokenizer = Tokenizer(cfg)
    entries = [
        {"id": "d_a", "tokens": ["apple", "banana"]},
        {"id": "d_b", "tokens": ["apple", "apple", "banana", "cherry"]},
        {"id": "d_c", "tokens": ["cherry", "date", "fig"]},
        {"id": "d_d", "tokens": ["banana"]},
    ]
    idx.add(entries)
    return idx, idx.get_view(), tokenizer


class TestTFIDFInputs:
    def test_string_query_tokenized(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        out = retriever.retrieve_candidates("apple", k=10)
        ids = [r["id"] for r in out]
        assert "d_a" in ids and "d_b" in ids

    def test_list_query_passed_through(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        out = retriever.retrieve_candidates(["apple"], k=10)
        assert any(r["id"] == "d_a" for r in out)

    def test_invalid_query_returns_empty(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        assert retriever.retrieve_candidates(42, k=5) == []

    @pytest.mark.parametrize("k", [0, -3])
    def test_non_positive_k_returns_empty(self, sparse_components, k: int) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        assert retriever.retrieve_candidates("apple", k=k) == []


class TestTFIDFScoring:
    def test_higher_tf_ranks_higher(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        out = retriever.retrieve_candidates("apple", k=10)
        scores = {r["id"]: r["score"] for r in out}
        assert scores["d_b"] > scores["d_a"]

    def test_sublinear_dampens_high_tf(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        sub = TFIDFRetriever(idx, view, tok, TFIDFConfig(sublinear_tf=True))
        lin = TFIDFRetriever(idx, view, tok, TFIDFConfig(sublinear_tf=False))
        sub_scores = {r["id"]: r["score"] for r in sub.retrieve_candidates("apple", k=10)}
        lin_scores = {r["id"]: r["score"] for r in lin.retrieve_candidates("apple", k=10)}
        # For d_b (TF=2): linear=2, sublinear=1+log(2)≈1.69 → linear is larger.
        assert lin_scores["d_b"] > sub_scores["d_b"]

    def test_results_sorted_desc_then_id(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        out = retriever.retrieve_candidates("apple banana cherry", k=10)
        keys = [(-r["score"], r["id"]) for r in out]
        assert keys == sorted(keys)

    def test_unknown_token_yields_no_results(self, sparse_components) -> None:
        idx, view, tok = sparse_components
        retriever = TFIDFRetriever(idx, view, tok, TFIDFConfig())
        assert retriever.retrieve_candidates("zzz", k=10) == []
