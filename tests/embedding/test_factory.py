"""Tests for the embedding registry and factory.

The factory wires together the registry, the cache, and contract validation.
We avoid importing concrete model providers (BGE-M3, Gemma) so tests run on
CPU-only environments without heavy ML dependencies.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from rag.embedding.base import BaseEmbedder
from rag.embedding.config import EmbeddingConfig
from rag.embedding.factory import (
    EmbeddingRegistry,
    create_embedder,
    get_default_registry,
)
from rag.embedding.model_cache import ModelCache


class _CapturedEmbedder(BaseEmbedder):
    """Test embedder that records the config and cache it received."""

    last_cache: Optional[ModelCache] = None
    last_config: Optional[EmbeddingConfig] = None

    def __init__(self, config: EmbeddingConfig, cache: Optional[ModelCache] = None) -> None:
        _CapturedEmbedder.last_config = config
        _CapturedEmbedder.last_cache = cache

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0]] * len(texts)

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [[0.0]] * len(texts)

    def dimension(self) -> int:
        return 1


def test_registry_register_and_get() -> None:
    registry = EmbeddingRegistry()
    registry.register("fake", _CapturedEmbedder)
    assert registry.get("fake") is _CapturedEmbedder


def test_registry_rejects_duplicate_registration() -> None:
    registry = EmbeddingRegistry()
    registry.register("x", _CapturedEmbedder)
    with pytest.raises(ValueError, match="already registered"):
        registry.register("x", _CapturedEmbedder)


def test_registry_unknown_provider_raises() -> None:
    registry = EmbeddingRegistry()
    registry.register("a", _CapturedEmbedder)
    with pytest.raises(ValueError, match="Unknown provider"):
        registry.get("missing")


def test_default_registry_has_known_providers() -> None:
    registry = get_default_registry()
    # The registry should know about bge-m3 and gemma even if the modules fail to import.
    assert "bge-m3" in registry._providers
    assert "gemma" in registry._providers


def test_create_embedder_wires_cache_and_validates_contract(tmp_path) -> None:
    registry = EmbeddingRegistry()
    registry.register("fake", _CapturedEmbedder)
    config = EmbeddingConfig(provider="fake", model_name="m1")
    cache = ModelCache(max_size=2)

    emb = create_embedder(config, cache=cache, registry=registry)
    assert isinstance(emb, _CapturedEmbedder)
    assert _CapturedEmbedder.last_cache is cache
    assert _CapturedEmbedder.last_config is config


def test_create_embedder_uses_default_cache_when_none() -> None:
    registry = EmbeddingRegistry()
    registry.register("fake", _CapturedEmbedder)
    config = EmbeddingConfig(provider="fake", model_name="m1")
    create_embedder(config, cache=None, registry=registry)
    assert _CapturedEmbedder.last_cache is not None


class _BrokenEmbedder(BaseEmbedder):
    """Declares dense but returns dim=None — caught by contract validation."""

    def __init__(self, config, cache=None):
        pass

    def embed_documents(self, texts):
        return [[] for _ in texts]

    def embed_queries(self, texts):
        return [[] for _ in texts]

    def dimension(self):
        return None


def test_create_embedder_runs_contract_validation() -> None:
    registry = EmbeddingRegistry()
    registry.register("broken", _BrokenEmbedder)
    cfg = EmbeddingConfig(provider="broken", model_name="x")
    with pytest.raises(ValueError, match="dimension=None"):
        create_embedder(cfg, registry=registry)
