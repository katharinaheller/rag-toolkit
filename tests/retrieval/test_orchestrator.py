"""Tests for RetrievalOrchestrator.

The orchestrator wires together dense and sparse retrievers, the document
store join, and the reranker. These tests cover dependency validation, mode
routing, empty-query handling and the retrieve_with_trace contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from rag.indexing.config import SparseIndexConfig
from rag.indexing.sparse.sparse_index import SparseIndex
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.indexing.store import DocumentStore
from rag.retrieval.config import (
    BM25Config,
    DenseRetrievalConfig,
    FusionConfig,
    LearnedSparseConfig,
    RerankerConfig,
    RetrievalConfig,
    TFIDFConfig,
)
from rag.retrieval.orchestrator import RetrievalOrchestrator
from tests.fixtures.factories import make_dense_vector, make_stored_document
from tests.mocks.embedders import FakeDenseEmbedder, FakeHybridEmbedder


class FakeDenseIndex:
    def __init__(self) -> None:
        self.added: List[dict] = []

    def add(self, entries):
        self.added.extend(entries)

    def query(self, query, k):
        return [{"id": e["id"], "score": float(i)} for i, e in enumerate(reversed(self.added))][:k]

    def persist(self) -> None:
        pass

    def load(self) -> None:
        pass


def _seed_store(tmp_path: Path) -> DocumentStore:
    store = DocumentStore(tmp_path / "docs.jsonl")
    docs = [
        make_stored_document(
            doc_id=f"emb_{i}", chunk_id=f"c_{i}", document_id=f"d_{i}",
            text=f"alpha beta gamma document {i}",
            dense=make_dense_vector(seed=i),
            sparse={"alpha": 1.0, "doc": 0.5},
            embedding_type="hybrid",
        )
        for i in range(3)
    ]
    store.write_many(docs)
    return store


def _seed_sparse_index(tmp_path: Path, store: DocumentStore) -> SparseIndex:
    cfg = SparseIndexConfig(tokenizer="simple")
    idx = SparseIndex(cfg, tmp_path / "sparse")
    tokenizer = Tokenizer(cfg)
    entries = [
        {"id": doc["id"], "tokens": tokenizer.tokenize(doc["text"])}
        for doc in store.stream_all()
    ]
    idx.add(entries)
    return idx


def _seed_dense_index(store: DocumentStore) -> FakeDenseIndex:
    idx = FakeDenseIndex()
    idx.add([{"id": d["id"], "dense_vector": d["dense_vector"]} for d in store.stream_all()])
    return idx


class TestDependencyValidation:
    def test_dense_mode_requires_dense_index(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(mode="dense")
        with pytest.raises(ValueError, match="dense_index"):
            RetrievalOrchestrator(
                document_store=store, config=cfg,
                embedder=FakeDenseEmbedder(), dense_index=None,
            )

    def test_dense_mode_requires_embedder(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(mode="dense")
        with pytest.raises(ValueError, match="embedder"):
            RetrievalOrchestrator(
                document_store=store, config=cfg,
                embedder=None, dense_index=_seed_dense_index(store),
            )

    def test_bm25_mode_requires_sparse_index_and_tokenizer(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(mode="sparse_bm25")
        with pytest.raises(ValueError, match="sparse_index"):
            RetrievalOrchestrator(
                document_store=store, config=cfg,
                tokenizer=Tokenizer(SparseIndexConfig()),
            )
        with pytest.raises(ValueError, match="tokenizer"):
            RetrievalOrchestrator(
                document_store=store, config=cfg,
                sparse_index=_seed_sparse_index(tmp_path, store),
            )

    def test_learned_sparse_requires_embedder(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(mode="sparse_learned")
        with pytest.raises(ValueError, match="embedder"):
            RetrievalOrchestrator(
                document_store=store, config=cfg, embedder=None,
            )


class TestEmptyQuery:
    @pytest.mark.parametrize("query", ["", "   ", "\t\n"])
    def test_blank_query_returns_empty(self, tmp_path: Path, query: str) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(
            mode="dense",
            dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            embedder=FakeDenseEmbedder(),
            dense_index=_seed_dense_index(store),
        )
        assert orch.retrieve(query, k=5) == []

    @pytest.mark.parametrize("k", [0, -1])
    def test_non_positive_k_returns_empty(self, tmp_path: Path, k: int) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(
            mode="dense",
            dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            embedder=FakeDenseEmbedder(),
            dense_index=_seed_dense_index(store),
        )
        assert orch.retrieve("query", k=k) == []


class TestDenseMode:
    def test_dense_mode_returns_joined_documents(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(
            mode="dense",
            dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            embedder=FakeDenseEmbedder(),
            dense_index=_seed_dense_index(store),
        )
        out = orch.retrieve("alpha", k=5)
        assert all("text" in r and "metadata" in r for r in out)
        assert all(r["score"] == r["retrieval_score"] for r in out)

    def test_default_k_uses_top_k(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(
            mode="dense", top_k=2,
            dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            embedder=FakeDenseEmbedder(),
            dense_index=_seed_dense_index(store),
        )
        out = orch.retrieve("alpha")
        assert len(out) <= 2


class TestSparseMode:
    def test_bm25_mode(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        sparse = _seed_sparse_index(tmp_path, store)
        cfg = RetrievalConfig(
            mode="sparse_bm25",
            bm25=BM25Config(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            sparse_index=sparse,
            tokenizer=Tokenizer(SparseIndexConfig()),
        )
        out = orch.retrieve("alpha", k=5)
        assert len(out) > 0

    def test_tfidf_mode(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        sparse = _seed_sparse_index(tmp_path, store)
        cfg = RetrievalConfig(
            mode="sparse_tfidf",
            tfidf=TFIDFConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            sparse_index=sparse,
            tokenizer=Tokenizer(SparseIndexConfig()),
        )
        out = orch.retrieve("alpha", k=5)
        assert len(out) > 0


class TestHybridMode:
    def test_hybrid_bm25_combines_both_legs(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        sparse = _seed_sparse_index(tmp_path, store)
        cfg = RetrievalConfig(
            mode="hybrid",
            sparse_strategy="bm25",
            dense=DenseRetrievalConfig(candidate_k=5),
            bm25=BM25Config(candidate_k=5),
            fusion=FusionConfig(strategy="weighted_sum"),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            embedder=FakeHybridEmbedder(),
            dense_index=_seed_dense_index(store),
            sparse_index=sparse,
            tokenizer=Tokenizer(SparseIndexConfig()),
        )
        out = orch.retrieve("alpha", k=5)
        assert len(out) > 0
        # Hybrid fusion preserves dense/sparse score metadata when applicable.
        for r in out:
            assert r["score"] == r["retrieval_score"]


class TestTrace:
    def test_trace_contains_intermediate_state(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        sparse = _seed_sparse_index(tmp_path, store)
        cfg = RetrievalConfig(
            mode="sparse_bm25",
            bm25=BM25Config(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            sparse_index=sparse,
            tokenizer=Tokenizer(SparseIndexConfig()),
        )
        trace = orch.retrieve_with_trace("alpha document", k=5)
        assert trace["mode"] == "sparse_bm25"
        assert trace["query_tokens"]
        assert trace["sparse_candidates"]
        assert "final_results" in trace

    def test_trace_for_blank_query(self, tmp_path: Path) -> None:
        store = _seed_store(tmp_path)
        cfg = RetrievalConfig(
            mode="dense",
            dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        orch = RetrievalOrchestrator(
            document_store=store, config=cfg,
            embedder=FakeDenseEmbedder(),
            dense_index=_seed_dense_index(store),
        )
        trace = orch.retrieve_with_trace("", k=5)
        assert trace["final_results"] == []
