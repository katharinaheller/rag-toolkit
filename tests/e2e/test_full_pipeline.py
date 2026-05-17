"""End-to-end pipeline test: ingest → embed → index → retrieve → generate.

This test cements the contract surface between every major component. The
LLM call is mocked at the `requests.post` boundary so the test stays offline
and deterministic while still exercising the real strategy, builder, client
and orchestrator code paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from rag.embedding.config import EmbeddingConfig
from rag.embedding.orchestrator import EmbeddingOrchestrator
from rag.generation.client import OllamaClient
from rag.generation.config import GenerationConfig
from rag.generation.context import ContextPreparer
from rag.generation.prompt_builder import STRICT_RAG_TEMPLATE, PromptBuilder
from rag.generation.strategies import SimpleRAGStrategy
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
from tests.mocks.embedders import FakeHybridEmbedder
from tests.mocks.ollama import install_requests_mock, ok_response


def _build_corpus(tmp_path: Path):
    cleaner = DefaultCleaner()
    chunker = SlidingWindowChunker(chunk_size=100, overlap=10, strategy=FIXED_OVERLAP)
    docs = [
        {"id": "doc_rag", "content":
         "Retrieval-Augmented Generation (RAG) combines retrieval and generation. "
         "RAG models retrieve relevant documents before generating answers.",
         "metadata": {"source": "rag.md"}},
        {"id": "doc_bm25", "content":
         "BM25 is a probabilistic ranking function used in information retrieval. "
         "It builds on TF-IDF and adds term saturation and length normalisation.",
         "metadata": {"source": "bm25.md"}},
        {"id": "doc_faiss", "content":
         "FAISS is a library for efficient similarity search and clustering of dense vectors. "
         "It supports flat, HNSW, and IVF index types.",
         "metadata": {"source": "faiss.md"}},
    ]
    chunks: List[dict] = []
    for raw in docs:
        cleaned = cleaner.clean(raw["content"])
        assert cleaned is not None
        doc = {"id": raw["id"], "content": cleaned, "metadata": raw["metadata"]}
        chunks.extend(chunker.chunk(doc))

    emb_cfg = EmbeddingConfig(provider="fake", model_name="fake-hybrid",
                              model_version="v1", batch_size=8,
                              retrieval_mode="hybrid")
    embedder = FakeHybridEmbedder(dim=24)
    embeddings = EmbeddingOrchestrator(embedder, emb_cfg).run(chunks)

    idx_cfg = IndexConfig(
        index_dir=tmp_path / "index", mode="hybrid",
        dense=DenseIndexConfig(backend="brute_force", metric="cosine"),
        sparse=SparseIndexConfig(tokenizer="simple"),
    )
    idx_result = IndexingOrchestrator(idx_cfg).build(embeddings=embeddings,
                                                     chunks=chunks)
    return idx_cfg, idx_result, embedder


def _build_strategy() -> SimpleRAGStrategy:
    gen_cfg = GenerationConfig(model_name="mistral", base_url="http://test:11434")
    client = OllamaClient(gen_cfg)
    builder = PromptBuilder(STRICT_RAG_TEMPLATE)
    preparer = ContextPreparer(max_context_chars=gen_cfg.max_context_chars)
    return SimpleRAGStrategy(client, builder, preparer)


class TestFullPipeline:
    def test_pipeline_produces_generated_answer(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        idx_cfg, idx_result, embedder = _build_corpus(tmp_path)
        ret_cfg = RetrievalConfig(
            mode="hybrid", sparse_strategy="bm25", top_k=3,
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
        strategy = _build_strategy()

        captured = install_requests_mock(
            monkeypatch, lambda url, **kw: ok_response("RAG retrieves and then generates.")
        )

        retrieved = retriever.retrieve("What is RAG?", k=3)
        assert retrieved
        result = strategy.generate("What is RAG?", [r["text"] for r in retrieved])

        assert result.success
        assert result.answer == "RAG retrieves and then generates."
        assert result.context_chars > 0
        assert result.prompt_chars > 0
        # One LLM call expected.
        assert len(captured) == 1

    def test_pipeline_is_deterministic(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        idx_cfg, idx_result, embedder = _build_corpus(tmp_path)
        ret_cfg = RetrievalConfig(
            mode="hybrid", sparse_strategy="bm25", top_k=3,
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
        install_requests_mock(monkeypatch, lambda url, **kw: ok_response("Same answer."))
        strategy = _build_strategy()

        a = retriever.retrieve("BM25 information retrieval", k=3)
        b = retriever.retrieve("BM25 information retrieval", k=3)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        assert [r["score"] for r in a] == [r["score"] for r in b]

        # Repeat generation with the same retrieved context.
        ra = strategy.generate("BM25?", [r["text"] for r in a])
        rb = strategy.generate("BM25?", [r["text"] for r in b])
        assert ra.answer == rb.answer
        assert ra.prompt == rb.prompt

    def test_empty_query_short_circuits(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        idx_cfg, idx_result, embedder = _build_corpus(tmp_path)
        ret_cfg = RetrievalConfig(
            mode="hybrid", sparse_strategy="bm25", top_k=3,
            dense=DenseRetrievalConfig(candidate_k=5),
            bm25=BM25Config(candidate_k=5),
            reranker=RerankerConfig(enabled=False),
        )
        retriever = RetrievalOrchestrator.from_index_result(
            idx_result, ret_cfg, embedder=embedder,
            tokenizer=Tokenizer(idx_cfg.sparse),
        )
        captured = install_requests_mock(monkeypatch, lambda url, **kw: ok_response("never called"))
        assert retriever.retrieve("", k=3) == []
        assert captured == []
