"""Tests for the IndexingOrchestrator joining embeddings and chunks."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.config import DenseIndexConfig, IndexConfig
from rag.indexing.orchestrator import IndexingOrchestrator
from tests.fixtures.factories import make_chunk, make_dense_vector


def _emb(chunk_id: str, doc_id: str = "doc_1") -> dict:
    return {
        "id": f"emb_{chunk_id}",
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "embedding": make_dense_vector(dim=4, seed=hash(chunk_id) % 100),
        "sparse_embedding": None,
        "embedding_type": "dense",
        "model_type": "test",
        "projection_method": "none",
        "metadata": {},
    }


def _chunk(chunk_id: str, doc_id: str = "doc_1", text: str | None = None) -> dict:
    return make_chunk(chunk_id, doc_id, text or f"text for {chunk_id}")


class TestBuildValidation:
    def test_missing_chunk_id_raises(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense",
                          dense=DenseIndexConfig(backend="brute_force"))
        orch = IndexingOrchestrator(cfg)
        with pytest.raises(ValueError, match="Chunk not found"):
            orch.build(embeddings=[_emb("missing")], chunks=[_chunk("c1")])

    def test_missing_chunk_keys_raise(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        bad = {"id": "c1", "document_id": "d", "text": "t"}  # no metadata
        with pytest.raises(ValueError, match="missing required fields"):
            orch.build(embeddings=[_emb("c1")], chunks=[bad])

    def test_missing_embedding_keys_raise(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        bad = {"id": "e", "chunk_id": "c"}  # missing document_id
        with pytest.raises(ValueError, match="missing required fields"):
            orch.build(embeddings=[bad], chunks=[_chunk("c")])

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        with pytest.raises(ValueError, match="No documents could be built"):
            orch.build(embeddings=[], chunks=[])

    def test_chunk_text_must_be_non_empty(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        bad = make_chunk("c", text="   ")
        with pytest.raises(ValueError, match="non-empty"):
            orch.build(embeddings=[_emb("c")], chunks=[bad])


class TestDenseMode:
    def test_builds_dense_index_with_correct_count(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        embs = [_emb(f"c{i}") for i in range(3)]
        chunks = [_chunk(f"c{i}") for i in range(3)]
        result = orch.build(embeddings=embs, chunks=chunks)
        assert result.mode == "dense"
        assert result.document_count == 3
        assert result.dense_index is not None
        assert result.sparse_index is None

    def test_persist_writes_dense_state(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        orch.build(
            embeddings=[_emb("c1")], chunks=[_chunk("c1")], persist=True,
        )
        assert (tmp_path / "dense" / "brute_force_entries.jsonl").exists()


class TestSparseMode:
    def test_builds_sparse_index(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="sparse")
        orch = IndexingOrchestrator(cfg)
        embs = [_emb("c1"), _emb("c2")]
        chunks = [_chunk("c1", text="first chunk text"), _chunk("c2", text="second chunk")]
        result = orch.build(embeddings=embs, chunks=chunks)
        assert result.sparse_index is not None
        assert result.dense_index is None


class TestHybridMode:
    def test_builds_both_indexes(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="hybrid")
        orch = IndexingOrchestrator(cfg)
        embs = [_emb("c1"), _emb("c2")]
        chunks = [_chunk("c1", text="lexical first"), _chunk("c2", text="lexical second")]
        result = orch.build(embeddings=embs, chunks=chunks)
        assert result.dense_index is not None
        assert result.sparse_index is not None


class TestDocumentStore:
    def test_store_always_written_even_without_persist(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        orch.build(embeddings=[_emb("c1")], chunks=[_chunk("c1")])
        store_path = tmp_path / "documents.jsonl"
        assert store_path.exists()
        assert store_path.stat().st_size > 0

    def test_store_carries_text_and_metadata(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        result = orch.build(
            embeddings=[_emb("c1")],
            chunks=[_chunk("c1", text="hello world")],
        )
        docs = result.document_store.load_all()
        assert docs[0]["text"] == "hello world"
        assert "metadata" in docs[0]


class TestLoad:
    def test_load_returns_indexing_result(self, tmp_path: Path) -> None:
        cfg = IndexConfig(index_dir=tmp_path, mode="dense")
        orch = IndexingOrchestrator(cfg)
        orch.build(embeddings=[_emb("c1")], chunks=[_chunk("c1")], persist=True)

        loader = IndexingOrchestrator(cfg)
        loaded = loader.load()
        assert loaded.dense_index is not None
        assert loaded.document_count == -1  # unknown after load
