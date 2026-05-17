"""Tests for the streaming EmbeddingOrchestrator.

Covered behaviours:
- Chunk validation
- Mode selection and dispatch
- Projection method resolution
- Normalisation and projection ordering
- Persistence
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.embedding.config import (
    EmbeddingBehaviorConfig,
    EmbeddingConfig,
    EmbeddingProjectionConfig,
)
from rag.embedding.orchestrator import EmbeddingOrchestrator
from rag.embedding.store import EmbeddingStore
from tests.fixtures.factories import make_chunk, make_chunks
from tests.mocks.embedders import FakeDenseEmbedder, FakeHybridEmbedder
from tests.utils.assertions import assert_unit_norm


@pytest.fixture
def dense_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="fake", model_name="fake-dense", batch_size=2,
        behavior=EmbeddingBehaviorConfig(normalize=False, mode="document"),
    )


@pytest.fixture
def normalising_config() -> EmbeddingConfig:
    return EmbeddingConfig(
        provider="fake", model_name="fake-dense", batch_size=4,
        behavior=EmbeddingBehaviorConfig(normalize=True, mode="document"),
    )


class TestModeSelection:
    def test_dense_mode_with_dense_only_embedder(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(dim=4), dense_config)
        results = orch.run([make_chunk()])
        assert results[0]["embedding_type"] == "dense"
        assert results[0]["embedding"] is not None
        assert results[0]["sparse_embedding"] is None

    def test_hybrid_mode_for_hybrid_embedder(self) -> None:
        cfg = EmbeddingConfig(provider="p", model_name="m", retrieval_mode="hybrid",
                              behavior=EmbeddingBehaviorConfig(normalize=False))
        orch = EmbeddingOrchestrator(FakeHybridEmbedder(dim=4), cfg)
        results = orch.run([make_chunk()])
        assert results[0]["embedding"] is not None
        assert results[0]["sparse_embedding"] is not None

    def test_sparse_mode_for_hybrid_embedder(self) -> None:
        cfg = EmbeddingConfig(provider="p", model_name="m", retrieval_mode="sparse",
                              behavior=EmbeddingBehaviorConfig(normalize=False))
        orch = EmbeddingOrchestrator(FakeHybridEmbedder(dim=4), cfg)
        results = orch.run([make_chunk()])
        assert results[0]["embedding"] is None
        assert results[0]["sparse_embedding"] is not None

    def test_dense_default_for_dense_embedder(self) -> None:
        cfg = EmbeddingConfig(provider="p", model_name="m",
                              behavior=EmbeddingBehaviorConfig(normalize=False))
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(dim=4), cfg)
        assert orch._retrieval_mode == "dense"

    def test_unsupported_mode_rejected_at_construction(self) -> None:
        cfg = EmbeddingConfig(provider="p", model_name="m", retrieval_mode="sparse",
                              behavior=EmbeddingBehaviorConfig(normalize=False))
        with pytest.raises(ValueError, match="not supported"):
            EmbeddingOrchestrator(FakeDenseEmbedder(), cfg)


class TestQueryMode:
    def test_query_mode_uses_query_methods(self) -> None:
        cfg = EmbeddingConfig(
            provider="p", model_name="m",
            behavior=EmbeddingBehaviorConfig(mode="query", normalize=False),
        )
        emb = FakeDenseEmbedder(dim=4)
        orch = EmbeddingOrchestrator(emb, cfg)
        chunk = make_chunk(text="query text")
        out = orch.run([chunk])
        expected = emb.embed_queries(["query text"])
        assert out[0]["embedding"] == expected[0]


class TestChunkValidation:
    def test_rejects_non_dict_chunk(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        with pytest.raises(ValueError, match="must be a dict"):
            orch.run(["not a dict"])

    def test_rejects_missing_required_keys(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        with pytest.raises(ValueError, match="missing required fields"):
            orch.run([{"id": "x", "text": "y"}])

    def test_rejects_non_string_text(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        bad = {"id": "x", "document_id": "d", "text": 42, "metadata": {}}
        with pytest.raises(ValueError, match="must be str"):
            orch.run([bad])

    def test_rejects_blank_text(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        bad = {"id": "x", "document_id": "d", "text": "   ", "metadata": {}}
        with pytest.raises(ValueError, match="empty or whitespace"):
            orch.run([bad])


class TestNormalisation:
    def test_normalize_true_yields_unit_vectors(self, normalising_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(dim=8), normalising_config)
        results = orch.run(make_chunks(3))
        for r in results:
            assert_unit_norm(r["embedding"])

    def test_normalize_false_preserves_raw_vectors(self, dense_config: EmbeddingConfig) -> None:
        emb = FakeDenseEmbedder(dim=4)
        orch = EmbeddingOrchestrator(emb, dense_config)
        chunk = make_chunk(text="text x")
        out = orch.run([chunk])
        expected = emb.embed_documents(["text x"])
        assert out[0]["embedding"] == expected[0]


class TestProjection:
    def test_resolves_to_none_when_target_dim_absent(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        assert orch._projection_method == "none"

    def test_resolves_to_mrl_for_mrl_model_when_model_aware(self) -> None:
        cfg = EmbeddingConfig(
            provider="p", model_name="m",
            behavior=EmbeddingBehaviorConfig(normalize=False),
            projection=EmbeddingProjectionConfig(target_dim=4, method="truncate", model_aware=True),
        )
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(dim=8, mrl=True), cfg)
        assert orch._projection_method == "mrl"

    def test_keeps_explicit_method_when_not_model_aware(self) -> None:
        cfg = EmbeddingConfig(
            provider="p", model_name="m",
            behavior=EmbeddingBehaviorConfig(normalize=False),
            projection=EmbeddingProjectionConfig(target_dim=4, method="truncate", model_aware=False),
        )
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(dim=8, mrl=True), cfg)
        assert orch._projection_method == "truncate"

    def test_projection_changes_output_dimension(self) -> None:
        cfg = EmbeddingConfig(
            provider="p", model_name="m",
            behavior=EmbeddingBehaviorConfig(normalize=False),
            projection=EmbeddingProjectionConfig(target_dim=4, method="truncate", model_aware=False),
        )
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(dim=8), cfg)
        out = orch.run([make_chunk()])
        assert len(out[0]["embedding"]) == 4

    def test_sparse_only_rejects_projection(self) -> None:
        cfg = EmbeddingConfig(
            provider="p", model_name="m", retrieval_mode="sparse",
            behavior=EmbeddingBehaviorConfig(normalize=False),
            projection=EmbeddingProjectionConfig(target_dim=128),
        )
        with pytest.raises(ValueError, match="Projection not applicable"):
            EmbeddingOrchestrator(FakeHybridEmbedder(), cfg)


class TestPersistence:
    def test_persist_requires_store(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        with pytest.raises(ValueError, match="persist=True requires a store"):
            orch.run([make_chunk()], persist=True)

    def test_persist_resets_store_then_writes(self, tmp_path: Path,
                                              dense_config: EmbeddingConfig) -> None:
        store_path = tmp_path / "emb.jsonl"
        store_path.write_text("stale\n", encoding="utf-8")
        store = EmbeddingStore(store_path)
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config, store=store)
        orch.run(make_chunks(3), persist=True)
        rows = store.read_all()
        assert len(rows) == 3

    def test_no_persist_leaves_store_empty(self, tmp_path: Path,
                                           dense_config: EmbeddingConfig) -> None:
        store = EmbeddingStore(tmp_path / "e.jsonl")
        store.reset()
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config, store=store)
        orch.run(make_chunks(2), persist=False)
        assert store.read_all() == []


class TestStreaming:
    def test_stream_yields_in_batches(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        chunks = make_chunks(5)
        out = list(orch.stream(chunks))
        assert len(out) == 5
        ids = [o["chunk_id"] for o in out]
        assert ids == [c["id"] for c in chunks]

    def test_run_returns_list_of_records(self, dense_config: EmbeddingConfig) -> None:
        orch = EmbeddingOrchestrator(FakeDenseEmbedder(), dense_config)
        out = orch.run(make_chunks(3))
        assert isinstance(out, list)
        assert all("embedding" in r for r in out)
