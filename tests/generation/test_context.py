"""Tests for context sanitisation and the ContextPreparer wrapper."""

from __future__ import annotations

import pytest

from rag.generation.context import ContextPreparer, sanitize_context


def test_passes_through_clean_strings() -> None:
    out = sanitize_context(["a", "b", "c"], max_context_chars=100)
    assert out == ["a", "b", "c"]


def test_strips_whitespace() -> None:
    out = sanitize_context(["  hello  ", "\nworld\n"], max_context_chars=100)
    assert out == ["hello", "world"]


def test_extracts_text_from_dicts() -> None:
    out = sanitize_context(
        [{"text": "first"}, {"text": "second"}], max_context_chars=100
    )
    assert out == ["first", "second"]


def test_mixed_strings_and_dicts() -> None:
    out = sanitize_context(["a", {"text": "b"}, "c"], max_context_chars=100)
    assert out == ["a", "b", "c"]


def test_skips_none_and_invalid_types() -> None:
    out = sanitize_context(["x", None, 42, {"text": "y"}], max_context_chars=100)
    assert out == ["x", "y"]


def test_skips_dict_with_non_string_text() -> None:
    out = sanitize_context([{"text": 123}, {"text": "ok"}], max_context_chars=100)
    assert out == ["ok"]


def test_skips_dict_without_text_key() -> None:
    out = sanitize_context([{"body": "no"}, "yes"], max_context_chars=100)
    assert out == ["yes"]


def test_deduplicates_preserving_first_occurrence_order() -> None:
    out = sanitize_context(["a", "b", "a", "c", "b"], max_context_chars=100)
    assert out == ["a", "b", "c"]


def test_skips_empty_strings_after_strip() -> None:
    out = sanitize_context(["   ", "", "x"], max_context_chars=100)
    assert out == ["x"]


def test_truncates_at_budget() -> None:
    out = sanitize_context(["a" * 50, "b" * 50, "c" * 50], max_context_chars=110)
    assert len(out) == 2


def test_hard_truncates_oversized_first_chunk() -> None:
    out = sanitize_context(["a" * 200], max_context_chars=100)
    assert len(out) == 1
    assert len(out[0]) == 100


def test_budget_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_context_chars must be"):
        sanitize_context(["x"], max_context_chars=0)


def test_empty_input_returns_empty_list() -> None:
    assert sanitize_context([], max_context_chars=100) == []


class TestContextPreparer:
    def test_init_requires_positive_budget(self) -> None:
        with pytest.raises(ValueError):
            ContextPreparer(max_context_chars=0)

    def test_max_context_chars_property(self) -> None:
        p = ContextPreparer(max_context_chars=42)
        assert p.max_context_chars == 42

    def test_prepare_delegates_to_sanitize_context(self) -> None:
        p = ContextPreparer(max_context_chars=20)
        out = p.prepare(["abc", "def", "abc"])
        assert out == ["abc", "def"]

    def test_prepare_truncates(self) -> None:
        p = ContextPreparer(max_context_chars=10)
        out = p.prepare(["1234567", "abcdef"])
        assert out == ["1234567"]
