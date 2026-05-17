from typing import Dict, Optional, Type

from rag.embedding.base import BaseEmbedder
from rag.embedding.config import EmbeddingConfig
from rag.embedding.model_cache import ModelCache, get_default_cache
from rag.embedding.models.bge_m3 import BGEM3Embedder
from rag.embedding.models.gemma import GemmaEmbedder


class EmbeddingRegistry:
    """Maps provider names to embedder classes."""

    def __init__(self) -> None:
        self._providers: Dict[str, Type[BaseEmbedder]] = {}

    def register(self, name: str, cls: Type[BaseEmbedder]) -> None:
        if name in self._providers:
            raise ValueError(f"Provider '{name}' already registered.")
        self._providers[name] = cls

    def get(self, name: str) -> Type[BaseEmbedder]:
        cls = self._providers.get(name)
        if cls is None:
            raise ValueError(
                f"Unknown provider: '{name}'. Supported: {sorted(self._providers)}."
            )
        return cls


def get_default_registry() -> EmbeddingRegistry:
    """Return a fresh registry with the built-in providers."""
    registry = EmbeddingRegistry()
    registry.register("bge-m3", BGEM3Embedder)
    registry.register("gemma", GemmaEmbedder)
    return registry


def create_embedder(
    config: EmbeddingConfig,
    cache: Optional[ModelCache] = None,
    registry: Optional[EmbeddingRegistry] = None,
) -> BaseEmbedder:
    """Instantiate an embedder and validate its contract.

    Contract validation runs immediately so capability mismatches are caught
    before the embedder reaches the orchestrator.
    """
    resolved_registry = registry or get_default_registry()
    cls = resolved_registry.get(config.provider)
    resolved_cache = cache or get_default_cache()
    embedder = cls(config, cache=resolved_cache)
    embedder.validate_embedding_contract()
    return embedder
