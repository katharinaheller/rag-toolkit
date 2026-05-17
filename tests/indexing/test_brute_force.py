"""Tests for the exact-search BruteForceIndex."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from rag.indexing.backends.brute_force import BruteForceIndex
from rag.indexing.config import DenseIndexConfig
from tests.utils.assertions import assert_sorted_desc_by_score_then_id


def _entries(*pairs):
    # Accept either splat tuples or a single iterable of tuples.
    if len(pairs) == 1 and not isinstance(pairs[0], tuple):
        pairs = tuple(pairs[0])
    return [{"id": rid, "dense_vector": v} for rid, v in pairs]


@pytest.fixture
def cos_index(tmp_path: Path) -> BruteForceIndex:
    return BruteForceIndex(DenseIndexConfig(backend="brute_force", metric="cosine"), tmp_path)


@pytest.fixture
def dot_index(tmp_path: Path) -> BruteForceIndex:
    return BruteForceIndex(DenseIndexConfig(backend="brute_force", metric="dot"), tmp_path)


@pytest.fixture
def l2_index(tmp_path: Path) -> BruteForceIndex:
    return BruteForceIndex(DenseIndexConfig(backend="brute_force", metric="l2"), tmp_path)


class TestAdd:
    def test_empty_batch_is_no_op(self, cos_index) -> None:
        cos_index.add([])
        assert cos_index.query([1.0, 0.0], k=1) == []

    def test_rejects_inconsistent_dimensions_within_batch(self, cos_index) -> None:
        with pytest.raises(ValueError, match="Inconsistent vector dimensions"):
            cos_index.add(_entries(("a", [1.0, 0.0]), ("b", [1.0])))

    def test_rejects_empty_vectors(self, cos_index) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            cos_index.add(_entries(("a", [])))

    def test_subsequent_batches_must_match_dimension(self, cos_index) -> None:
        cos_index.add(_entries(("a", [1.0, 0.0])))
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cos_index.add(_entries(("b", [1.0, 0.0, 0.0])))

    def test_explicit_dimension_must_match(self, tmp_path: Path) -> None:
        idx = BruteForceIndex(
            DenseIndexConfig(backend="brute_force", metric="cosine", dimension=3), tmp_path,
        )
        with pytest.raises(ValueError, match="match config.dimension"):
            idx.add(_entries(("a", [1.0, 0.0])))


class TestQuery:
    def test_returns_empty_for_empty_index(self, cos_index) -> None:
        assert cos_index.query([1.0, 0.0], k=5) == []

    def test_returns_empty_for_k_zero(self, cos_index) -> None:
        cos_index.add(_entries(("a", [1.0, 0.0])))
        assert cos_index.query([1.0, 0.0], k=0) == []

    def test_returns_empty_for_empty_query(self, cos_index) -> None:
        cos_index.add(_entries(("a", [1.0, 0.0])))
        assert cos_index.query([], k=5) == []

    def test_rejects_non_list_query(self, cos_index) -> None:
        cos_index.add(_entries(("a", [1.0, 0.0])))
        with pytest.raises(TypeError):
            cos_index.query("not a list", k=1)  # type: ignore[arg-type]

    def test_cosine_returns_high_for_aligned_vectors(self, cos_index) -> None:
        cos_index.add(_entries(
            ("a", [1.0, 0.0]),
            ("b", [0.0, 1.0]),
            ("c", [1.0, 1.0]),
        ))
        results = cos_index.query([1.0, 0.0], k=3)
        assert results[0]["id"] == "a"
        assert math.isclose(results[0]["score"], 1.0, abs_tol=1e-6)

    def test_dot_product_unbounded(self, dot_index) -> None:
        dot_index.add(_entries(("a", [2.0, 2.0]), ("b", [0.5, 0.5])))
        results = dot_index.query([2.0, 2.0], k=2)
        assert results[0]["id"] == "a"
        assert results[0]["score"] > results[1]["score"]

    def test_l2_returns_negative_distance(self, l2_index) -> None:
        l2_index.add(_entries(("near", [1.0, 0.0]), ("far", [10.0, 10.0])))
        results = l2_index.query([1.0, 0.0], k=2)
        # near should be at the top with score = 0 (negated distance)
        assert results[0]["id"] == "near"
        assert results[0]["score"] > results[1]["score"]

    def test_deterministic_tiebreak_by_id_ascending(self, cos_index) -> None:
        # Same vector → same score → tiebreak by id ascending.
        cos_index.add(_entries(
            ("c", [1.0, 0.0]),
            ("a", [1.0, 0.0]),
            ("b", [1.0, 0.0]),
        ))
        results = cos_index.query([1.0, 0.0], k=3)
        assert [r["id"] for r in results] == ["a", "b", "c"]

    def test_results_sorted_desc_score_asc_id(self, cos_index) -> None:
        cos_index.add(_entries(
            ("d", [0.5, 0.5]),
            ("a", [1.0, 0.0]),
            ("b", [0.9, 0.1]),
        ))
        results = cos_index.query([1.0, 0.0], k=3)
        assert_sorted_desc_by_score_then_id(results)

    def test_top_k_truncation(self, cos_index) -> None:
        cos_index.add(_entries(
            (f"id_{i}", [1.0 / (i + 1), 0.0]) for i in range(5)
        ))
        assert len(cos_index.query([1.0, 0.0], k=3)) == 3

    def test_query_dim_mismatch_raises(self, cos_index) -> None:
        cos_index.add(_entries(("a", [1.0, 0.0])))
        with pytest.raises(ValueError, match="dim"):
            cos_index.query([1.0, 0.0, 0.0], k=1)


class TestPersistence:
    def test_persist_then_load_round_trip(self, tmp_path: Path) -> None:
        idx = BruteForceIndex(DenseIndexConfig(metric="cosine"), tmp_path)
        idx.add(_entries(("a", [1.0, 0.0]), ("b", [0.0, 1.0])))
        idx.persist()

        loaded = BruteForceIndex(DenseIndexConfig(metric="cosine"), tmp_path)
        loaded.load()
        results = loaded.query([1.0, 0.0], k=2)
        ids = {r["id"] for r in results}
        assert ids == {"a", "b"}

    def test_load_missing_file_is_no_op(self, tmp_path: Path) -> None:
        idx = BruteForceIndex(DenseIndexConfig(metric="cosine"), tmp_path)
        idx.load()
        assert idx.query([1.0, 0.0], k=5) == []
