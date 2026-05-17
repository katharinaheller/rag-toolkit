"""Deterministic fake embedders for unit and integration tests.

These implement the BaseEmbedder contract using simple, predictable functions
so that test results never depend on heavy ML dependencies or random seeds
inside third-party libraries.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Optional, Set, Tuple

from rag.embedding.base import BaseEmbedder


def _seeded_vector(text: str, dim: int, salt: str = "") -> List[float]:
    """Produce a deterministic dense vector from text content."""
    digest = hashlib.sha256((salt + "|" + text).encode("utf-8")).digest()
    vec = []
    while len(vec) < dim:
        for b in digest:
            vec.append((b - 128) / 128.0)
            if len(vec) >= dim:
                break
        digest = hashlib.sha256(digest).digest()
    return vec[:dim]


def _seeded_sparse(text: str) -> Dict[str, float]:
    """Produce a small deterministic sparse weight dict from text content."""
    tokens = text.lower().split()
    sparse: Dict[str, float] = {}
    for tok in tokens:
        h = hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8]
        sparse[h] = sparse.get(h, 0.0) + 1.0
    return sparse


class FakeDenseEmbedder(BaseEmbedder):
    """Deterministic dense-only embedder used in tests."""

    def __init__(self, dim: int = 8, model_name: str = "fake-dense", mrl: bool = False) -> None:
        self._dim = dim
        self._model_name = model_name
        self._mrl = mrl

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_seeded_vector(t, self._dim, salt="doc") for t in texts]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [_seeded_vector(t, self._dim, salt="query") for t in texts]

    def dimension(self) -> Optional[int]:
        return self._dim

    def supported_modes(self) -> Set[str]:
        return {"dense"}

    def default_retrieval_mode(self) -> str:
        return "dense"

    def is_mrl_model(self) -> bool:
        return self._mrl

    def model_type(self) -> str:
        return self._model_name


class FakeHybridEmbedder(BaseEmbedder):
    """Hybrid embedder producing dense and sparse outputs deterministically."""

    def __init__(self, dim: int = 8, model_name: str = "fake-hybrid") -> None:
        self._dim = dim
        self._model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [_seeded_vector(t, self._dim, salt="doc") for t in texts]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [_seeded_vector(t, self._dim, salt="query") for t in texts]

    def embed_documents_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        return [_seeded_sparse(t) for t in texts]

    def embed_queries_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        return [_seeded_sparse(t) for t in texts]

    def embed_documents_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        return self.embed_documents(texts), self.embed_documents_sparse(texts)

    def embed_queries_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        return self.embed_queries(texts), self.embed_queries_sparse(texts)

    def dimension(self) -> Optional[int]:
        return self._dim

    def supported_modes(self) -> Set[str]:
        return {"dense", "sparse", "hybrid"}

    def default_retrieval_mode(self) -> str:
        return "hybrid"

    def model_type(self) -> str:
        return self._model_name


class FakeFailingEmbedder(BaseEmbedder):
    """Embedder whose query methods raise — used to test fault tolerance."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.1] * 4 for _ in texts]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        raise RuntimeError("synthetic dense failure")

    def embed_queries_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        raise NotImplementedError("not supported")

    def dimension(self) -> Optional[int]:
        return 4
