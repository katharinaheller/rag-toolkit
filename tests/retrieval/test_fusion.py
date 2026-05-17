"""Tests for the score-fusion module.

Validates both fusion strategies (weighted_sum, rrf), candidate combination
modes (union, intersection), normalisation behaviour and deterministic order.
"""

from __future__ import annotations

import pytest

from rag.retrieval.config import FusionConfig
from rag.retrieval.fusion import fuse


def _dense():
    return [
        {"id": "a", "score": 0.9},
        {"id": "b", "score": 0.5},
        {"id": "c", "score": 0.1},
    ]


def _sparse():
    return [
        {"id": "b", "score": 4.0},
        {"id": "c", "score": 2.0},
        {"id": "d", "score": 1.0},
    ]


class TestEdgeCases:
    def test_k_zero_returns_empty(self) -> None:
        assert fuse(_dense(), _sparse(), FusionConfig(), k=0) == []

    def test_empty_inputs_return_empty(self) -> None:
        assert fuse([], [], FusionConfig(), k=5) == []

    def test_intersection_with_disjoint_returns_empty(self) -> None:
        cfg = FusionConfig(combination="intersection")
        out = fuse(
            [{"id": "x", "score": 1.0}],
            [{"id": "y", "score": 1.0}],
            cfg, k=5,
        )
        assert out == []


class TestWeightedSum:
    def test_union_includes_all_ids(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum", combination="union",
                           dense_weight=0.5, sparse_weight=0.5)
        out = fuse(_dense(), _sparse(), cfg, k=10)
        assert {r["id"] for r in out} == {"a", "b", "c", "d"}

    def test_intersection_only_shared_ids(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum", combination="intersection",
                           dense_weight=0.5, sparse_weight=0.5)
        out = fuse(_dense(), _sparse(), cfg, k=10)
        assert {r["id"] for r in out} == {"b", "c"}

    def test_results_sorted_desc_then_id(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum")
        out = fuse(_dense(), _sparse(), cfg, k=10)
        keys = [(-r["score"], r["id"]) for r in out]
        assert keys == sorted(keys)

    def test_normalize_brings_scores_into_unit_range(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum", normalize_scores=True,
                           dense_weight=1.0, sparse_weight=0.0)
        out = fuse(_dense(), _sparse(), cfg, k=10)
        dense_scores = [r["score"] for r in out if r["dense_score"] is not None]
        assert max(dense_scores) <= 1.0 + 1e-9

    def test_raw_scores_preserved_alongside_combined(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum")
        out = fuse(_dense(), _sparse(), cfg, k=10)
        for r in out:
            assert "dense_score" in r
            assert "sparse_score" in r

    def test_dense_weight_zero_uses_only_sparse(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum", dense_weight=0.0,
                           sparse_weight=1.0, normalize_scores=False)
        out = fuse(_dense(), _sparse(), cfg, k=10)
        scored = {r["id"]: r["score"] for r in out}
        assert scored["d"] == pytest.approx(1.0)
        assert scored["a"] == pytest.approx(0.0)

    def test_weights_proportional_contribution(self) -> None:
        cfg = FusionConfig(strategy="weighted_sum", dense_weight=1.0,
                           sparse_weight=1.0, normalize_scores=False)
        out = fuse(_dense(), _sparse(), cfg, k=10)
        scored = {r["id"]: r["score"] for r in out}
        # "b" appears in both lists: 0.5 + 4.0 = 4.5
        assert scored["b"] == pytest.approx(4.5)


class TestRRF:
    def test_rrf_uses_rank_not_score(self) -> None:
        cfg = FusionConfig(strategy="rrf", rrf_k=60)
        out = fuse(_dense(), _sparse(), cfg, k=10)
        scored = {r["id"]: r["score"] for r in out}
        # "a" is dense rank 1, absent from sparse: 1/(60+1) + 1/(60+4)
        expected_a = 1.0 / 61.0 + 1.0 / 64.0
        assert scored["a"] == pytest.approx(expected_a)
        # "b": dense rank 2, sparse rank 1: 1/62 + 1/61
        expected_b = 1.0 / 62.0 + 1.0 / 61.0
        assert scored["b"] == pytest.approx(expected_b)

    def test_rrf_intersection_only(self) -> None:
        cfg = FusionConfig(strategy="rrf", combination="intersection")
        out = fuse(_dense(), _sparse(), cfg, k=10)
        assert {r["id"] for r in out} == {"b", "c"}

    def test_rrf_truncates_to_k(self) -> None:
        cfg = FusionConfig(strategy="rrf")
        out = fuse(_dense(), _sparse(), cfg, k=2)
        assert len(out) == 2

    def test_rrf_returns_dense_and_sparse_metadata(self) -> None:
        cfg = FusionConfig(strategy="rrf")
        out = fuse(_dense(), _sparse(), cfg, k=10)
        for r in out:
            assert "dense_score" in r and "sparse_score" in r
