"""Integration: embedding → indexing.

Joins EmbeddingVector dicts with chunks and routes them through the real
IndexingOrchestrator. Validates DocumentStore persistence and dense index
querying on the BruteForceIndex backend.
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
from rag.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from rag.ingestion.chunking.strategies import FIXED_OVERLAP
from rag.ingestion.cleaner import DefaultCleaner
from tests.mocks.embedders import FakeDenseEmbedder


def _build_docs_and_embeddings(tmp_path: Path):
    cleaner = DefaultCleaner()
    chunker = SlidingWindowChunker(chunk_size=40, overlap=5, strategy=FIXED_OVERLAP)
    raw_docs = [
        {"id": "doc_a", "content": "Alpha document discussing apples.", "metadata": {"source": "a.md"}},
        {"id": "doc_b", "content": "Beta document discussing bananas.", "metadata": {"source": "b.md"}},
        {"id": "doc_c", "content": "Cherries grow on cherry trees.", "metadata": {"source": "c.md"}},
    ]
    all_chunks: List[dict] = []
    for raw in raw_docs:
        cleaned = cleaner.clean(raw["content"])
        assert cleaned is not None
        doc = {"id": raw["id"], "content": cleaned, "metadata": raw["metadata"]}
        all_chunks.extend(chunker.chunk(doc))

    cfg = EmbeddingConfig(provider="fake", model_name="fake", model_version="v1", batch_size=8)
    embedder = FakeDenseEmbedder(dim=16)
    embeddings = EmbeddingOrchestrator(embedder, cfg).run(all_chunks)
    return all_chunks, embeddings


class TestEmbeddingToIndexing:
    def test_dense_mode_builds_index_and_document_store(self, tmp_path: Path) -> None:
        chunks, embeddings = _build_docs_and_embeddings(tmp_path)

        idx_cfg = IndexConfig(
            index_dir=tmp_path / "index",
            mode="dense",
            dense=DenseIndexConfig(backend="brute_force", metric="cosine"),
        )
        orch = IndexingOrchestrator(idx_cfg)
        result = orch.build(embeddings=embeddings, chunks=chunks, persist=False)

        assert result.mode == "dense"
        assert result.document_count == len(chunks)
        assert result.dense_index is not None
        # Query the index with one of the dense vectors.
        first_vec = embeddings[0]["embedding"]
        out = result.dense_index.query(first_vec, k=3)
        assert out
        assert out[0]["id"] == embeddings[0]["id"]

    def test_hybrid_mode_builds_both_indexes(self, tmp_path: Path) -> None:
        chunks, embeddings = _build_docs_and_embeddings(tmp_path)

        idx_cfg = IndexConfig(
            index_dir=tmp_path / "index",
            mode="hybrid",
            dense=DenseIndexConfig(backend="brute_force", metric="cosine"),
            sparse=SparseIndexConfig(tokenizer="simple"),
        )
        orch = IndexingOrchestrator(idx_cfg)
        result = orch.build(embeddings=embeddings, chunks=chunks, persist=True)

        assert result.dense_index is not None
        assert result.sparse_index is not None

    def test_missing_chunk_raises_value_error(self, tmp_path: Path) -> None:
        chunks, embeddings = _build_docs_and_embeddings(tmp_path)
        broken_chunks = chunks[1:]  # drop the chunk referenced by embeddings[0]

        idx_cfg = IndexConfig(index_dir=tmp_path / "index", mode="dense")
        orch = IndexingOrchestrator(idx_cfg)
        with pytest.raises(ValueError, match="Chunk not found"):
            orch.build(embeddings=embeddings, chunks=broken_chunks)

    def test_document_store_persisted_to_disk(self, tmp_path: Path) -> None:
        chunks, embeddings = _build_docs_and_embeddings(tmp_path)
        idx_cfg = IndexConfig(index_dir=tmp_path / "index", mode="dense")
        IndexingOrchestrator(idx_cfg).build(
            embeddings=embeddings, chunks=chunks, persist=False,
        )
        assert (tmp_path / "index" / "documents.jsonl").exists()
