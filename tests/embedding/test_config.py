"""Validation tests for the immutable embedding configuration classes."""

from __future__ import annotations

import pytest

from rag.embedding.config import (
    EmbeddingBehaviorConfig,
    EmbeddingConfig,
    EmbeddingProjectionConfig,
)


class TestEmbeddingBehaviorConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = EmbeddingBehaviorConfig()
        assert cfg.normalize is True
        assert cfg.mode == "document"
        assert cfg.query_prefix is None
        assert cfg.document_prefix is None

    @pytest.mark.parametrize("mode", ["document", "query"])
    def test_accepts_valid_modes(self, mode: str) -> None:
        EmbeddingBehaviorConfig(mode=mode)

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode must be one of"):
            EmbeddingBehaviorConfig(mode="hybrid")

    def test_is_frozen(self) -> None:
        cfg = EmbeddingBehaviorConfig()
        with pytest.raises(Exception):
            cfg.mode = "query"  # type: ignore[misc]


class TestEmbeddingProjectionConfig:
    def test_defaults(self) -> None:
        cfg = EmbeddingProjectionConfig()
        assert cfg.target_dim is None
        assert cfg.method == "truncate"
        assert cfg.model_aware is True

    @pytest.mark.parametrize("method", ["truncate", "mrl", "pad"])
    def test_accepts_valid_methods(self, method: str) -> None:
        EmbeddingProjectionConfig(method=method)

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="method must be one of"):
            EmbeddingProjectionConfig(method="quantize")

    @pytest.mark.parametrize("bad_dim", [0, -1, -16])
    def test_rejects_non_positive_target_dim(self, bad_dim: int) -> None:
        with pytest.raises(ValueError, match="target_dim must be positive"):
            EmbeddingProjectionConfig(target_dim=bad_dim)


class TestEmbeddingConfig:
    def test_minimum_valid_config(self) -> None:
        cfg = EmbeddingConfig(provider="bge-m3", model_name="BAAI/bge-m3")
        assert cfg.batch_size == 32
        assert cfg.device == "cpu"

    @pytest.mark.parametrize("bs", [0, -1, -10])
    def test_rejects_non_positive_batch_size(self, bs: int) -> None:
        with pytest.raises(ValueError, match="batch_size must be positive"):
            EmbeddingConfig(provider="p", model_name="m", batch_size=bs)

    @pytest.mark.parametrize("seq", [0, -32])
    def test_rejects_non_positive_max_seq_length(self, seq: int) -> None:
        with pytest.raises(ValueError, match="max_seq_length must be positive"):
            EmbeddingConfig(provider="p", model_name="m", max_seq_length=seq)

    def test_max_seq_length_none_is_allowed(self) -> None:
        EmbeddingConfig(provider="p", model_name="m", max_seq_length=None)

    def test_dtype_flags_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            EmbeddingConfig(
                provider="p", model_name="m", use_bfloat16=True, use_fp16=True
            )

    @pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid", None])
    def test_accepts_valid_retrieval_modes(self, mode):
        EmbeddingConfig(provider="p", model_name="m", retrieval_mode=mode)

    def test_rejects_invalid_retrieval_mode(self) -> None:
        with pytest.raises(ValueError, match="retrieval_mode must be one of"):
            EmbeddingConfig(provider="p", model_name="m", retrieval_mode="bm25")
