"""Tests for generation utility helpers."""

from __future__ import annotations

import pytest

from rag.generation.models import GenerationResult
from rag.generation.utils import extract_texts, format_result_summary, truncate_text


class TestExtractTexts:
    def test_extracts_from_strings(self) -> None:
        assert extract_texts(["a", "b"]) == ["a", "b"]

    def test_extracts_from_dicts(self) -> None:
        assert extract_texts([{"text": "x"}, {"text": "y"}]) == ["x", "y"]

    def test_mixes_strings_and_dicts(self) -> None:
        assert extract_texts(["a", {"text": "b"}]) == ["a", "b"]

    def test_skips_dicts_without_text(self) -> None:
        assert extract_texts([{"text": "ok"}, {"body": "skip"}]) == ["ok"]

    def test_skips_non_string_text_value(self) -> None:
        assert extract_texts([{"text": 123}, "ok"]) == ["ok"]

    def test_empty_list(self) -> None:
        assert extract_texts([]) == []


class TestTruncateText:
    def test_truncates_long_string(self) -> None:
        assert truncate_text("hello world", 5) == "hello"

    def test_returns_short_string_intact(self) -> None:
        assert truncate_text("hi", 100) == "hi"

    def test_zero_max_chars_returns_empty(self) -> None:
        assert truncate_text("hello", 0) == ""

    def test_rejects_negative_max_chars(self) -> None:
        with pytest.raises(ValueError):
            truncate_text("x", -1)


class TestFormatSummary:
    def test_includes_essential_fields(self) -> None:
        r = GenerationResult(
            answer="short", prompt="p", model="m", strategy="s",
            template_name="t", template_version="v", latency_ms=42.0,
            prompt_chars=10, context_chars=5,
        )
        out = format_result_summary(r)
        assert "Strategy" in out and "s" in out
        assert "Latency" in out and "42" in out
        assert "Answer" in out
        assert "Error" not in out

    def test_truncates_long_answers(self) -> None:
        r = GenerationResult(answer="x" * 500, prompt="", model="", strategy="",
                             template_name="", template_version="")
        out = format_result_summary(r)
        assert "…" in out

    def test_includes_error_when_present(self) -> None:
        r = GenerationResult(answer="", prompt="", model="", strategy="",
                             template_name="", template_version="", error="oops")
        assert "Error" in format_result_summary(r)
