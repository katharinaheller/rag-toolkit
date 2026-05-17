"""Tests for the deterministic Tokenizer used in indexing and retrieval."""

from __future__ import annotations

import pytest

from rag.indexing.config import SparseIndexConfig
from rag.indexing.sparse.tokenizer import Tokenizer


class TestSimpleTokenizer:
    def setup_method(self) -> None:
        self.tok = Tokenizer(SparseIndexConfig(tokenizer="simple"))

    def test_lowercases(self) -> None:
        assert self.tok.tokenize("Hello WORLD") == ["hello", "world"]

    def test_strips_punctuation(self) -> None:
        assert self.tok.tokenize("Hello, world!") == ["hello", "world"]

    def test_treats_numbers_as_tokens(self) -> None:
        assert self.tok.tokenize("Mistral 7B model") == ["mistral", "7b", "model"]

    def test_unicode_word_characters(self) -> None:
        out = self.tok.tokenize("naïve résumé")
        assert "naïve" in out
        assert "résumé" in out

    def test_empty_input(self) -> None:
        assert self.tok.tokenize("") == []


class TestWhitespaceTokenizer:
    def setup_method(self) -> None:
        self.tok = Tokenizer(SparseIndexConfig(tokenizer="whitespace"))

    def test_preserves_punctuation(self) -> None:
        assert self.tok.tokenize("hello, world!") == ["hello,", "world!"]

    def test_lowercases(self) -> None:
        assert self.tok.tokenize("Hello World") == ["hello", "world"]


def test_deterministic_across_calls() -> None:
    tok = Tokenizer(SparseIndexConfig(tokenizer="simple"))
    text = "The quick brown fox jumps over the lazy dog"
    a = tok.tokenize(text)
    b = tok.tokenize(text)
    assert a == b
