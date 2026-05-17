"""Tests for vector arithmetic and validation helpers."""

from __future__ import annotations

import math

import pytest

from rag.indexing.utils import (
    cosine_similarity,
    dot_product,
    l2_distance,
    l2_normalize,
    validate_real_vector,
)


def test_cosine_identical_vectors_is_one() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0, abs_tol=1e-9)


def test_cosine_orthogonal_vectors_is_zero() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0, abs_tol=1e-9)


def test_cosine_opposite_vectors_is_negative_one() -> None:
    assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0, abs_tol=1e-9)


def test_cosine_zero_vector_yields_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_dot_product_basic() -> None:
    assert dot_product([1.0, 2.0], [3.0, 4.0]) == 11.0


def test_l2_distance_basic() -> None:
    assert math.isclose(l2_distance([0.0, 0.0], [3.0, 4.0]), 5.0, abs_tol=1e-9)


def test_l2_normalize_produces_unit_norm() -> None:
    out = l2_normalize([3.0, 4.0])
    assert math.isclose(sum(x * x for x in out), 1.0, abs_tol=1e-9)


def test_l2_normalize_zero_vector_unchanged() -> None:
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]


class TestValidateRealVector:
    def test_accepts_floats(self) -> None:
        validate_real_vector([1.0, -2.5, 0.0], "v")

    def test_accepts_ints(self) -> None:
        validate_real_vector([1, -3, 0], "v")

    def test_rejects_booleans(self) -> None:
        with pytest.raises(TypeError):
            validate_real_vector([True, False], "v")  # type: ignore[list-item]

    def test_rejects_strings(self) -> None:
        with pytest.raises(TypeError):
            validate_real_vector([1.0, "x"], "v")  # type: ignore[list-item]

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            validate_real_vector([1.0, float("nan")], "v")

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError):
            validate_real_vector([float("inf")], "v")
