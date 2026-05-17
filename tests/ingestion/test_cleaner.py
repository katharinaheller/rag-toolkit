"""Tests for DefaultCleaner.

The cleaner is a critical normalisation step: it must be deterministic and
strip exactly the documented set of artefacts (BOM, CRLF, trailing whitespace,
trailing blank lines) and return None for content that is empty after
cleaning. These tests pin every documented behaviour.
"""

from __future__ import annotations

import pytest

from rag.ingestion.cleaner import Cleaner, DefaultCleaner


@pytest.fixture
def cleaner() -> DefaultCleaner:
    return DefaultCleaner()


class TestProtocolConformance:
    def test_default_cleaner_satisfies_cleaner_protocol(
        self, cleaner: DefaultCleaner,
    ) -> None:
        assert isinstance(cleaner, Cleaner)


class TestBOMStripping:
    def test_bom_at_start_removed(self, cleaner: DefaultCleaner) -> None:
        text = "\ufeffhello world"
        assert cleaner.clean(text) == "hello world"

    def test_bom_only_at_start(self, cleaner: DefaultCleaner) -> None:
        # BOM appearing mid-text is not stripped.
        text = "hello\ufeffworld"
        assert cleaner.clean(text) == "hello\ufeffworld"


class TestLineEndings:
    @pytest.mark.parametrize("source,expected", [
        ("a\r\nb", "a\nb"),
        ("a\rb", "a\nb"),
        ("a\r\nb\r\nc", "a\nb\nc"),
        ("a\nb", "a\nb"),
    ])
    def test_line_endings_normalized(
        self, cleaner: DefaultCleaner, source: str, expected: str,
    ) -> None:
        assert cleaner.clean(source) == expected


class TestTrailingWhitespace:
    def test_trailing_spaces_removed_per_line(self, cleaner: DefaultCleaner) -> None:
        assert cleaner.clean("line one   \nline two\t\t") == "line one\nline two"

    def test_internal_spaces_preserved(self, cleaner: DefaultCleaner) -> None:
        assert cleaner.clean("hello  world") == "hello  world"


class TestTrailingBlankLines:
    def test_strip_trailing_blank_lines(self, cleaner: DefaultCleaner) -> None:
        assert cleaner.clean("text\n\n\n") == "text"

    def test_keep_internal_blank_lines(self, cleaner: DefaultCleaner) -> None:
        assert cleaner.clean("para1\n\npara2") == "para1\n\npara2"


class TestEmptyOutputBehavior:
    @pytest.mark.parametrize("source", ["", "   ", "\n\n\n", "\t\r\n", "\ufeff"])
    def test_blank_returns_none(self, cleaner: DefaultCleaner, source: str) -> None:
        assert cleaner.clean(source) is None


class TestDeterminism:
    def test_repeated_call_yields_same_output(self, cleaner: DefaultCleaner) -> None:
        text = "\ufeffalpha\r\n\nbeta\t\n\n\n"
        first = cleaner.clean(text)
        for _ in range(3):
            assert cleaner.clean(text) == first


class TestUnicode:
    def test_preserves_non_ascii(self, cleaner: DefaultCleaner) -> None:
        text = "café — résumé\n"
        assert cleaner.clean(text) == "café — résumé"
