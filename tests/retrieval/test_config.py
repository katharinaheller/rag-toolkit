"""Validation tests for retrieval configuration classes."""

from __future__ import annotations

import pytest

from rag.retrieval.config import (
    BM25Config,
    DenseRetrievalConfig,
    FusionConfig,
    LearnedSparseConfig,
    RerankerConfig,
    RetrievalConfig,
    TFIDFConfig,
)


class TestDenseRetrievalConfig:
    def test_default(self) -> None:
        assert DenseRetrievalConfig().candidate_k == 100

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive(self, k: int) -> None:
        with pytest.raises(ValueError, match="candidate_k"):
            DenseRetrievalConfig(candidate_k=k)


class TestBM25Config:
    def test_defaults(self) -> None:
        c = BM25Config()
        assert c.k1 == 1.5
        assert c.b == 0.75

    def test_rejects_negative_k1(self) -> None:
        with pytest.raises(ValueError, match="k1"):
            BM25Config(k1=-0.1)

    @pytest.mark.parametrize("b", [-0.1, 1.1])
    def test_rejects_b_out_of_unit_range(self, b: float) -> None:
        with pytest.raises(ValueError, match="b must be in"):
            BM25Config(b=b)


class TestTFIDFConfig:
    def test_defaults(self) -> None:
        c = TFIDFConfig()
        assert c.sublinear_tf is True

    def test_rejects_non_positive_candidate_k(self) -> None:
        with pytest.raises(ValueError):
            TFIDFConfig(candidate_k=0)


class TestLearnedSparseConfig:
    def test_defaults(self) -> None:
        c = LearnedSparseConfig()
        assert c.use_index_pruning is True

    def test_rejects_non_positive_candidate_k(self) -> None:
        with pytest.raises(ValueError):
            LearnedSparseConfig(candidate_k=0)

    def test_rejects_negative_min_score(self) -> None:
        with pytest.raises(ValueError):
            LearnedSparseConfig(min_score=-0.1)


class TestFusionConfig:
    @pytest.mark.parametrize("s", ["weighted_sum", "rrf"])
    def test_accepts_valid_strategies(self, s: str) -> None:
        FusionConfig(strategy=s)

    def test_rejects_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="strategy"):
            FusionConfig(strategy="ranknet")

    @pytest.mark.parametrize("c", ["union", "intersection"])
    def test_accepts_combination_modes(self, c: str) -> None:
        FusionConfig(combination=c)

    def test_rejects_invalid_combination(self) -> None:
        with pytest.raises(ValueError, match="combination"):
            FusionConfig(combination="mean")

    @pytest.mark.parametrize("w", [(-0.1, 0.5), (0.5, -0.1)])
    def test_rejects_negative_weights(self, w: tuple) -> None:
        with pytest.raises(ValueError, match="weights"):
            FusionConfig(dense_weight=w[0], sparse_weight=w[1])

    def test_rejects_non_positive_rrf_k(self) -> None:
        with pytest.raises(ValueError, match="rrf_k"):
            FusionConfig(rrf_k=0)


class TestRerankerConfig:
    @pytest.mark.parametrize("w", [-0.1, 1.1])
    def test_rejects_out_of_range_lexical_weight(self, w: float) -> None:
        with pytest.raises(ValueError, match="lexical_weight"):
            RerankerConfig(lexical_weight=w)

    @pytest.mark.parametrize("w", [-0.1, 1.1])
    def test_rejects_out_of_range_length_weight(self, w: float) -> None:
        with pytest.raises(ValueError, match="length_weight"):
            RerankerConfig(length_weight=w)

    def test_rejects_non_positive_target_length(self) -> None:
        with pytest.raises(ValueError, match="target_length"):
            RerankerConfig(target_length=0)


class TestRetrievalConfig:
    @pytest.mark.parametrize("mode", [
        "dense", "sparse_bm25", "sparse_tfidf", "sparse_learned", "hybrid",
    ])
    def test_accepts_valid_modes(self, mode: str) -> None:
        RetrievalConfig(mode=mode)

    def test_rejects_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            RetrievalConfig(mode="something")

    def test_rejects_non_positive_top_k(self) -> None:
        with pytest.raises(ValueError, match="top_k"):
            RetrievalConfig(top_k=0)
