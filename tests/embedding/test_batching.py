"""Tests for the streaming batch iterator."""

from __future__ import annotations

import pytest

from rag.embedding.batching import batch_iter


def test_batch_iter_yields_correct_sizes() -> None:
    batches = list(batch_iter(range(10), batch_size=3))
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    assert sum(batches, []) == list(range(10))


def test_batch_iter_yields_empty_for_empty_input() -> None:
    assert list(batch_iter([], batch_size=4)) == []


def test_batch_iter_exact_multiple_no_short_tail() -> None:
    batches = list(batch_iter(range(6), batch_size=2))
    assert [len(b) for b in batches] == [2, 2, 2]


def test_batch_iter_does_not_materialise_input() -> None:
    """Each yielded batch must be available before the next is constructed."""
    produced = []

    def gen():
        for i in range(5):
            produced.append(i)
            yield i

    it = batch_iter(gen(), batch_size=2)
    first = next(it)
    assert first == [0, 1]
    assert produced == [0, 1]
    next(it)
    assert produced == [0, 1, 2, 3]


@pytest.mark.parametrize("invalid_size", [0, -1, -100])
def test_batch_iter_rejects_non_positive_batch_size(invalid_size: int) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(batch_iter(range(3), batch_size=invalid_size))


def test_batch_iter_preserves_element_order() -> None:
    src = ["a", "b", "c", "d", "e"]
    flat = [item for batch in batch_iter(src, batch_size=2) for item in batch]
    assert flat == src
