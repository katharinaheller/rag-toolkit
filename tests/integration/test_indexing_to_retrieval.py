"""Integration: indexing → retrieval.

The retrieval orchestrator must operate end-to-end against indexes built by
the indexing orchestrator. This exercises the contract that connects them
(document store join, ScoredCandidate→RetrievalResult, ordering).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from rag.embedding.config import EmbeddingConfig
from rag.embedding.orchestrator import EmbeddingOrchestrator
from rag.indexing.config import (
    DenseIndexConfig,
    IndexConfig,
    SparseIndexConfig,
)
from rag.indexing.orchestrator import IndexingOrchestrator
from rag.indexing.sparse.tokenizer import Tokenizer
from rag.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from rag.ingestion.chunking.strategies import FIXED_OVERLAP
from rag.ingestion.cleaner import DefaultCleaner
from rag.retrieval.config import (
    BM25Config,
    DenseRetrievalConfig,
    FusionConfig,
    RerankerConfig,
    RetrievalConfig,
)
from rag.retrieval.orchestrator import RetrievalOrchestrator
from tests.mocks.embedders import FakeDenseEmbedder, FakeHybridEmbedder


def _build_index(tmp_path: Path, mode: str = "hybrid"):
    cleaner = DefaultCleaner()
    chunker = SlidingWindowChunker(80, 10, FIXED_OVERLAP)

    raw_docs = [
        {"id": "doc_apple", "content": "apple banana apple pie recipe with sugar", "metadata": {"source": "a.md"}},
        {"id": "doc_banana", "content": "banana split dessert with vanilla ice", "metadata": {"source": "b.md"}},
        {"id": "doc_cherry", "content": "cherry pie filling using fresh cherries", "metadata": {"source": "c.md"}},
        {"id": "doc_date", "content": "date palm fruit harvest season notes", "metadata": {"source": "d.md"}},
    ]

    all_chunks: List[dict] = []
    for raw in raw_docs:
        cleaned = cleaner.clean(raw["content"])
        assert cleaned is not None
        doc = {"id": raw["id"], "content": cleaned, "metadata": raw["metadata"]}
        all_chunks.extend(chunker.chunk(doc))

    emb_cfg = EmbeddingConfig(provider="fake", model_name="fake", model_version="v1",
                              batch_size=8, retrieval_mode="hybrid" if mode == "hybrid" else None)
    embedder = FakeHybridEmbedder(dim=16) if mode == "hybrid" else FakeDenseEmbedder(dim=16)
    embeddings = EmbeddingOrchestrator(embedder, emb_cfg).run(all_chunks)

    idx_cfg = IndexConfig(
        index_dir=tmp_path / "index", mode=mode,
        dense=DenseIndexConfig(backend="brute_force", metric="cosine"),
        sparse=SparseIndexConfig(tokenizer="simple"),
    )
    idx_orch = IndexingOrchestrator(idx_cfg)
    idx_result = idx_orch.build(embeddings=embeddings, chunks=all_chunks)

    return idx_cfg, idx_result, embedder


class TestRetrievalAgainstBuiltIndex:
    def test_dense_mode_end_to_end(self, tmp_path: Path) -> None:
        idx_cfg, idx_result, embedder = _build_index(tmp_path, mode="dense")
        ret_cfg = RetrievalConfig(
            mode="dense",
            dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
            top_k=3,
        )
        retriever = RetrievalOrchestrator.from_index_result(
            idx_result, ret_cfg, embedder=embedder,
        )
        out = retriever.retrieve("apple", k=3)
        assert out and len(out) <= 3
        assert all("text" in r for r in out)

    def test_sparse_bm25_returns_topical_match(self, tmp_path: Path) -> None:
        idx_cfg, idx_result, embedder = _build_index(tmp_path, mode="hybrid")
        ret_cfg = RetrievalConfig(
            mode="sparse_bm25",
            bm25=BM25Config(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        retriever = RetrievalOrchestrator.from_index_result(
            idx_result, ret_cfg, embedder=embedder,
            tokenizer=Tokenizer(idx_cfg.sparse),
        )
        out = retriever.retrieve("cherry", k=3)
        assert out
        # Document about cherries should be the top result.
        assert "cherry" in out[0]["text"].lower()

    def test_hybrid_mode_fuses_signals(self, tmp_path: Path) -> None:
        idx_cfg, idx_result, embedder = _build_index(tmp_path, mode="hybrid")
        ret_cfg = RetrievalConfig(
            mode="hybrid", sparse_strategy="bm25",
            dense=DenseRetrievalConfig(candidate_k=5),
            bm25=BM25Config(candidate_k=5),
            fusion=FusionConfig(strategy="weighted_sum",
                                normalize_scores=True),
            reranker=RerankerConfig(enabled=False),
        )
        retriever = RetrievalOrchestrator.from_index_result(
            idx_result, ret_cfg, embedder=embedder,
            tokenizer=Tokenizer(idx_cfg.sparse),
        )
        out = retriever.retrieve("banana", k=3)
        assert out
        # Results should be deterministically ordered.
        keys = [(-r["score"], r["id"]) for r in out]
        assert keys == sorted(keys)

    def test_empty_query_yields_empty(self, tmp_path: Path) -> None:
        idx_cfg, idx_result, embedder = _build_index(tmp_path, mode="dense")
        ret_cfg = RetrievalConfig(
            mode="dense", dense=DenseRetrievalConfig(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        retriever = RetrievalOrchestrator.from_index_result(
            idx_result, ret_cfg, embedder=embedder,
        )
        assert retriever.retrieve("", k=3) == []
