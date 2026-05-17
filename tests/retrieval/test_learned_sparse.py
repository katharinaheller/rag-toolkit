"""Tests for LearnedSparseRetriever.

Exercises both execution paths (pruned and full scan), the min_score filter,
empty input handling and ordering invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.store import DocumentStore
from rag.retrieval.config import LearnedSparseConfig
from rag.retrieval.learned_sparse import LearnedSparseRetriever
from tests.fixtures.factories import make_stored_document


@pytest.fixture
def doc_store(tmp_path: Path) -> DocumentStore:
    """Document store seeded with sparse-vector documents."""
    store = DocumentStore(tmp_path / "docs.jsonl")
    docs = [
        make_stored_document(
            doc_id="emb_a", chunk_id="c_a", document_id="d_a",
            text="alpha", dense=None,
            sparse={"t1": 1.0, "t2": 0.5}, embedding_type="sparse",
        ),
        make_stored_document(
            doc_id="emb_b", chunk_id="c_b", document_id="d_b",
            text="beta", dense=None,
            sparse={"t2": 0.7, "t3": 0.3}, embedding_type="sparse",
        ),
        make_stored_document(
            doc_id="emb_c", chunk_id="c_c", document_id="d_c",
            text="gamma", dense=None,
            sparse={"t3": 0.1, "t4": 0.9}, embedding_type="sparse",
        ),
        make_stored_document(
            doc_id="emb_no_sparse", chunk_id="c_n", document_id="d_n",
            text="dense only", dense=[0.1, 0.2], sparse=None,
            embedding_type="dense",
        ),
    ]
    store.write_many(docs)
    return store


class TestQueryShape:
    def test_non_dict_query_returns_empty(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(doc_store, LearnedSparseConfig())
        assert retriever.retrieve_candidates([1, 2], k=5) == []
        assert retriever.retrieve_candidates("string", k=5) == []

    def test_empty_dict_returns_empty(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(doc_store, LearnedSparseConfig())
        assert retriever.retrieve_candidates({}, k=5) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_returns_empty(self, doc_store: DocumentStore, k: int) -> None:
        retriever = LearnedSparseRetriever(doc_store, LearnedSparseConfig())
        assert retriever.retrieve_candidates({"t1": 1.0}, k=k) == []


class TestPrunedPath:
    def test_pruned_path_returns_matching_docs(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(
            doc_store, LearnedSparseConfig(use_index_pruning=True)
        )
        out = retriever.retrieve_candidates({"t1": 1.0}, k=10)
        ids = [r["id"] for r in out]
        assert "emb_a" in ids
        assert "emb_b" not in ids

    def test_pruned_path_dot_product_scoring(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(
            doc_store, LearnedSparseConfig(use_index_pruning=True)
        )
        out = retriever.retrieve_candidates({"t2": 1.0}, k=10)
        scores = {r["id"]: r["score"] for r in out}
        assert scores["emb_a"] == pytest.approx(0.5)
        assert scores["emb_b"] == pytest.approx(0.7)


class TestFullScanPath:
    def test_full_scan_returns_same_results_as_pruned(
        self, doc_store: DocumentStore,
    ) -> None:
        pruned = LearnedSparseRetriever(
            doc_store, LearnedSparseConfig(use_index_pruning=True)
        )
        scan = LearnedSparseRetriever(
            doc_store, LearnedSparseConfig(use_index_pruning=False)
        )
        q = {"t2": 1.0, "t3": 1.0}
        a = pruned.retrieve_candidates(q, k=10)
        b = scan.retrieve_candidates(q, k=10)
        assert [(r["id"], round(r["score"], 6)) for r in a] == \
               [(r["id"], round(r["score"], 6)) for r in b]


class TestFiltering:
    def test_min_score_filter_drops_low_results(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(
            doc_store, LearnedSparseConfig(min_score=0.6, use_index_pruning=False),
        )
        out = retriever.retrieve_candidates({"t2": 1.0}, k=10)
        assert all(r["score"] >= 0.6 for r in out)
        assert any(r["id"] == "emb_b" for r in out)
        assert not any(r["id"] == "emb_a" for r in out)

    def test_non_positive_scores_filtered_out(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(
            doc_store, LearnedSparseConfig(use_index_pruning=False),
        )
        out = retriever.retrieve_candidates({"t_unknown": 1.0}, k=10)
        assert out == []

    def test_docs_without_sparse_vector_skipped(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(doc_store, LearnedSparseConfig())
        out = retriever.retrieve_candidates({"t1": 1.0, "t2": 1.0}, k=10)
        assert not any(r["id"] == "emb_no_sparse" for r in out)


class TestOrdering:
    def test_ordering_by_score_desc_then_id(self, doc_store: DocumentStore) -> None:
        retriever = LearnedSparseRetriever(doc_store, LearnedSparseConfig())
        out = retriever.retrieve_candidates({"t2": 1.0, "t3": 1.0, "t4": 1.0}, k=10)
        keys = [(-r["score"], r["id"]) for r in out]
        assert keys == sorted(keys)
