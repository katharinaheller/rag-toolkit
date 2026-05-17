"""Tests for pure scoring helpers used across retrieval strategies."""

from __future__ import annotations

import math

import pytest

from rag.retrieval.scoring import (
    bm25_idf,
    bm25_score_document,
    bm25_tf_component,
    cosine_similarity,
    dot_product,
    min_max_normalise,
    sparse_dot_product,
    tfidf_idf,
    tfidf_score_document,
    tfidf_tf,
)


class TestCosineDot:
    def test_cosine_identical(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_cosine_zero_inputs(self) -> None:
        assert cosine_similarity([], []) == 0.0
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_dot_product(self) -> None:
        assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0

    def test_dot_product_handles_empty(self) -> None:
        assert dot_product([], []) == 0.0


class TestBM25:
    def test_idf_decreases_with_higher_df(self) -> None:
        a = bm25_idf(df=1, n=1000)
        b = bm25_idf(df=100, n=1000)
        assert a > b

    def test_idf_handles_zero_n(self) -> None:
        assert bm25_idf(df=0, n=0) == 0.0

    def test_tf_component_saturates_with_high_tf(self) -> None:
        low = bm25_tf_component(1, 100, 100.0, 1.5, 0.75)
        high = bm25_tf_component(100, 100, 100.0, 1.5, 0.75)
        # high TF should be larger but bounded
        assert high > low
        assert high < 100  # saturation prevents linear growth

    def test_tf_component_zero_for_zero_tf(self) -> None:
        assert bm25_tf_component(0, 10, 10.0, 1.5, 0.75) == 0.0

    def test_score_document_unknown_token_contributes_nothing(self) -> None:
        score = bm25_score_document(
            query_tokens=["unknown"], doc_id="d",
            doc_length=10, avg_doc_length=10.0, n_docs=10,
            document_frequency={}, get_tf_fn=lambda *_: 0,
            k1=1.5, b=0.75,
        )
        assert score == 0.0

    def test_score_document_increases_with_tf(self) -> None:
        df_map = {"x": 1}
        get_tf_low = lambda t, d: 1
        get_tf_high = lambda t, d: 5
        low = bm25_score_document(["x"], "d", 10, 10.0, 100, df_map, get_tf_low, 1.5, 0.75)
        high = bm25_score_document(["x"], "d", 10, 10.0, 100, df_map, get_tf_high, 1.5, 0.75)
        assert high > low

    def test_score_unique_query_tokens_only(self) -> None:
        df_map = {"x": 1}
        get_tf = lambda t, d: 1
        once = bm25_score_document(["x"], "d", 10, 10.0, 100, df_map, get_tf, 1.5, 0.75)
        twice = bm25_score_document(["x", "x"], "d", 10, 10.0, 100, df_map, get_tf, 1.5, 0.75)
        assert math.isclose(once, twice)


class TestTFIDF:
    def test_tf_zero_when_no_occurrence(self) -> None:
        assert tfidf_tf(0, sublinear=True) == 0.0
        assert tfidf_tf(0, sublinear=False) == 0.0

    def test_tf_sublinear_grows_slower_than_linear(self) -> None:
        sub = tfidf_tf(100, sublinear=True)
        lin = tfidf_tf(100, sublinear=False)
        assert sub < lin

    def test_idf_decreases_with_df(self) -> None:
        a = tfidf_idf(df=1, n=1000)
        b = tfidf_idf(df=100, n=1000)
        assert a > b

    def test_idf_returns_one_for_zero_n(self) -> None:
        assert tfidf_idf(df=0, n=0) == 1.0

    def test_score_document_zero_for_unknown_tokens(self) -> None:
        assert tfidf_score_document(
            ["unknown"], "d", 100, {}, lambda *_: 0, True
        ) == 0.0


class TestSparseDotProduct:
    def test_basic(self) -> None:
        a = {"t1": 1.0, "t2": 2.0}
        b = {"t1": 3.0, "t3": 4.0}
        assert sparse_dot_product(a, b) == 3.0

    def test_empty_returns_zero(self) -> None:
        assert sparse_dot_product({}, {"a": 1.0}) == 0.0
        assert sparse_dot_product({"a": 1.0}, {}) == 0.0

    def test_uses_shorter_dict_for_iteration(self) -> None:
        """No observable behaviour change — sanity check on commutativity."""
        a = {"t1": 1.0, "t2": 2.0, "t3": 3.0}
        b = {"t1": 4.0}
        assert sparse_dot_product(a, b) == sparse_dot_product(b, a)


class TestMinMaxNormalise:
    def test_empty(self) -> None:
        assert min_max_normalise([]) == []

    def test_equal_scores_all_one(self) -> None:
        assert min_max_normalise([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]

    def test_normalises_to_unit_range(self) -> None:
        out = min_max_normalise([1.0, 3.0, 5.0])
        assert out[0] == 0.0
        assert out[2] == 1.0
        assert math.isclose(out[1], 0.5)
