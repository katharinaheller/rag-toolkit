"""Tests for dimensionality projection utilities."""

from __future__ import annotations

import pytest

from rag.embedding.projection import project_batch, project_vector


def test_no_op_when_target_is_none() -> None:
    v = [1.0, 2.0, 3.0]
    assert project_vector(v, target_dim=None, method="truncate") is v


def test_no_op_when_target_equals_current_dim() -> None:
    v = [1.0, 2.0, 3.0]
    assert project_vector(v, target_dim=3, method="truncate") is v


def test_truncate_slices_prefix() -> None:
    v = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = project_vector(v, target_dim=3, method="truncate")
    assert out == [1.0, 2.0, 3.0]


def test_mrl_slices_prefix_for_smaller_target() -> None:
    v = list(range(1, 11))
    out = project_vector([float(x) for x in v], target_dim=4, method="mrl")
    assert out == [1.0, 2.0, 3.0, 4.0]


def test_pad_extends_to_larger_target() -> None:
    v = [1.0, 2.0]
    out = project_vector(v, target_dim=5, method="pad")
    assert out == [1.0, 2.0, 0.0, 0.0, 0.0]


def test_extending_without_pad_raises() -> None:
    with pytest.raises(ValueError, match="exceeds vector dimension"):
        project_vector([1.0, 2.0], target_dim=5, method="truncate")


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown projection method"):
        project_vector([1.0, 2.0, 3.0], target_dim=2, method="quantize")


def test_project_batch_applies_to_every_vector() -> None:
    batch = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    out = project_batch(batch, target_dim=2, method="truncate")
    assert out == [[1.0, 2.0], [5.0, 6.0]]


def test_project_batch_empty() -> None:
    assert project_batch([], target_dim=4, method="truncate") == []


def test_pad_to_smaller_still_truncates() -> None:
    """When target < current_dim, all methods including 'pad' truncate."""
    out = project_vector([1.0, 2.0, 3.0], target_dim=2, method="pad")
    assert out == [1.0, 2.0]
