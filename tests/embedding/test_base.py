"""Behavioural tests for the BaseEmbedder abstract contract.

These tests exercise the concrete defaults inherited by every embedder and
verify that the contract validator catches inconsistent capability claims.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import pytest

from rag.embedding.base import VALID_RETRIEVAL_MODES, BaseEmbedder


class _DenseOnly(BaseEmbedder):
    """Minimal dense-only embedder used to exercise base-class defaults."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]

    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        return [[float(len(t))] for t in texts]

    def dimension(self) -> Optional[int]:
        return 1


class _LiesAboutDense(BaseEmbedder):
    """Declares dense support but returns None — contract violation."""

    def embed_documents(self, texts):
        return [[] for _ in texts]

    def embed_queries(self, texts):
        return [[] for _ in texts]

    def dimension(self):
        return None

    def supported_modes(self) -> Set[str]:
        return {"dense"}


def test_valid_retrieval_modes_contains_expected_values() -> None:
    assert VALID_RETRIEVAL_MODES == frozenset({"dense", "sparse", "hybrid"})


def test_default_supported_modes_is_dense_only() -> None:
    emb = _DenseOnly()
    assert emb.supported_modes() == {"dense"}


def test_default_retrieval_mode_is_dense() -> None:
    assert _DenseOnly().default_retrieval_mode() == "dense"


def test_is_mrl_model_false_by_default() -> None:
    assert _DenseOnly().is_mrl_model() is False


def test_model_type_uses_lowercased_class_name() -> None:
    assert _DenseOnly().model_type() == "_denseonly"


def test_sparse_methods_raise_by_default() -> None:
    emb = _DenseOnly()
    with pytest.raises(NotImplementedError):
        emb.embed_documents_sparse(["x"])
    with pytest.raises(NotImplementedError):
        emb.embed_queries_sparse(["x"])


def test_hybrid_default_uses_both_dense_and_sparse() -> None:
    """Default hybrid pulls dense then sparse; sparse path should propagate."""
    emb = _DenseOnly()
    with pytest.raises(NotImplementedError):
        emb.embed_documents_hybrid(["x"])
    with pytest.raises(NotImplementedError):
        emb.embed_queries_hybrid(["x"])


def test_validate_contract_accepts_consistent_declarations() -> None:
    _DenseOnly().validate_embedding_contract()  # must not raise


def test_validate_contract_rejects_dense_without_dimension() -> None:
    with pytest.raises(ValueError, match="dimension=None"):
        _LiesAboutDense().validate_embedding_contract()


def test_validate_projection_config_default_noop() -> None:
    _DenseOnly().validate_projection_config(
        target_dim=None, resolved_method="none", original_method="truncate", model_aware=True
    )
