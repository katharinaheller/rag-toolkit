"""Integration: ingestion → embedding.

Exercises the boundary where chunked documents become embedding inputs.
Uses the real DefaultCleaner, SlidingWindowChunker, and EmbeddingOrchestrator
with a deterministic FakeDenseEmbedder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from rag.embedding.config import (
    EmbeddingBehaviorConfig,
    EmbeddingConfig,
    EmbeddingProjectionConfig,
)
from rag.embedding.orchestrator import EmbeddingOrchestrator
from rag.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from rag.ingestion.chunking.strategies import FIXED_OVERLAP
from rag.ingestion.cleaner import DefaultCleaner
from rag.ingestion.schema import Document
from tests.mocks.embedders import FakeDenseEmbedder
from tests.utils.assertions import assert_unit_norm


def _document(text: str, source: str = "f.md") -> Document:
    return {"id": "doc_x", "content": text, "metadata": {"source": source}}


@pytest.fixture
def cleaned_doc() -> Document:
    raw = "\ufeffAlpha document.\r\n\nSecond paragraph with content.   \n\n\n"
    cleaned = DefaultCleaner().clean(raw)
    assert cleaned is not None
    return {"id": "doc_x", "content": cleaned, "metadata": {"source": "f.md"}}


@pytest.fixture
def chunker() -> SlidingWindowChunker:
    return SlidingWindowChunker(chunk_size=25, overlap=5, strategy=FIXED_OVERLAP)


@pytest.fixture
def embed_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="fake", model_name="fake-dense", model_version="v1",
        batch_size=4,
        behavior=EmbeddingBehaviorConfig(normalize=True, mode="document"),
        projection=EmbeddingProjectionConfig(target_dim=None),
    )


class TestIngestionToEmbedding:
    def test_chunks_embed_successfully(
        self,
        cleaned_doc: Document,
        chunker: SlidingWindowChunker,
        embed_config: EmbeddingConfig,
    ) -> None:
        chunks = list(chunker.chunk(cleaned_doc))
        assert chunks

        embedder = FakeDenseEmbedder(dim=16)
        orch = EmbeddingOrchestrator(embedder=embedder, config=embed_config)
        vectors = orch.run(chunks)

        assert len(vectors) == len(chunks)
        for v in vectors:
            assert v["embedding"] is not None
            assert v["embedding_type"] == "dense"
            assert_unit_norm(v["embedding"])

    def test_chunk_ids_propagated_into_embeddings(
        self, cleaned_doc, chunker, embed_config,
    ) -> None:
        chunks = list(chunker.chunk(cleaned_doc))
        embedder = FakeDenseEmbedder(dim=8)
        orch = EmbeddingOrchestrator(embedder=embedder, config=embed_config)
        vectors = orch.run(chunks)
        for chunk, vector in zip(chunks, vectors):
            assert vector["chunk_id"] == chunk["id"]
            assert vector["document_id"] == chunk["document_id"]

    def test_repeated_run_yields_identical_vectors(
        self, cleaned_doc, chunker, embed_config,
    ) -> None:
        chunks = list(chunker.chunk(cleaned_doc))
        embedder_a = FakeDenseEmbedder(dim=8)
        embedder_b = FakeDenseEmbedder(dim=8)
        orch_a = EmbeddingOrchestrator(embedder=embedder_a, config=embed_config)
        orch_b = EmbeddingOrchestrator(embedder=embedder_b, config=embed_config)
        a = orch_a.run(chunks)
        b = orch_b.run(chunks)
        for va, vb in zip(a, b):
            assert va["embedding"] == vb["embedding"]
            assert va["id"] == vb["id"]
