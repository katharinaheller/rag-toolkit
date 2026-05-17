"""Tests for L2 normalization utilities."""

from __future__ import annotations

import math

import pytest

from rag.embedding.normalization import l2_normalize, normalize_batch
from tests.utils.assertions import assert_unit_norm


def test_l2_normalize_produces_unit_vector() -> None:
    out = l2_normalize([3.0, 4.0])
    assert_unit_norm(out)
    assert out == pytest.approx([0.6, 0.8])


def test_l2_normalize_zero_vector_returned_unchanged() -> None:
    zero = [0.0, 0.0, 0.0]
    assert l2_normalize(zero) == zero


def test_l2_normalize_preserves_dimension() -> None:
    out = l2_normalize([1.0] * 16)
    assert len(out) == 16
    assert_unit_norm(out)


def test_normalize_batch_normalizes_every_vector() -> None:
    batch = [[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]]
    out = normalize_batch(batch)
    assert_unit_norm(out[0])
    assert_unit_norm(out[1])
    assert out[2] == [0.0, 0.0]


def test_normalize_batch_empty_input() -> None:
    assert normalize_batch([]) == []


@pytest.mark.parametrize("v", [[1e-10, 1e-10], [-1.0, 2.0, -3.0]])
def test_normalized_vector_has_norm_close_to_one(v: list) -> None:
    out = l2_normalize(v)
    norm = math.sqrt(sum(x * x for x in out))
    assert math.isclose(norm, 1.0, abs_tol=1e-6)
