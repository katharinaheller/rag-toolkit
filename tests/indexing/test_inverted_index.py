"""Tests for the lexical InvertedIndex and its read-only view."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.sparse.inverted_index import InvertedIndex
from rag.indexing.sparse.view import InvertedIndexView


def _entries(*pairs):
    return [{"id": rid, "tokens": toks} for rid, toks in pairs]


@pytest.fixture
def index_with_corpus() -> InvertedIndex:
    idx = InvertedIndex()
    idx._build(_entries(
        ("d1", ["rag", "is", "a", "hybrid", "system"]),
        ("d2", ["bm25", "is", "a", "lexical", "scoring", "function"]),
        ("d3", ["hybrid", "search", "combines", "lexical", "and", "dense"]),
    ))
    return idx


class TestBuild:
    def test_build_records_doc_lengths(self, index_with_corpus) -> None:
        stats = index_with_corpus._get_stats()
        assert stats["doc_count"] == 3
        assert stats["doc_lengths"]["d1"] == 5

    def test_build_records_term_frequency(self, index_with_corpus) -> None:
        assert index_with_corpus._get_tf("is", "d1") == 1
        assert index_with_corpus._get_tf("missing", "d1") == 0

    def test_build_records_document_frequency(self, index_with_corpus) -> None:
        stats = index_with_corpus._get_stats()
        assert stats["document_frequency"]["is"] == 2
        assert stats["document_frequency"]["hybrid"] == 2

    def test_average_doc_length_correct(self, index_with_corpus) -> None:
        stats = index_with_corpus._get_stats()
        expected = (5 + 6 + 6) / 3
        assert abs(stats["avg_doc_length"] - expected) < 1e-9

    def test_repeated_tokens_count_once_per_document(self) -> None:
        idx = InvertedIndex()
        idx._build(_entries(("d1", ["x", "x", "y", "x"])))
        stats = idx._get_stats()
        assert idx._get_tf("x", "d1") == 3
        assert stats["document_frequency"]["x"] == 1

    def test_duplicate_id_raises(self) -> None:
        idx = InvertedIndex()
        idx._build(_entries(("d1", ["a"])))
        with pytest.raises(ValueError, match="Duplicate"):
            idx._build(_entries(("d1", ["b"])))

    def test_blank_id_raises(self) -> None:
        idx = InvertedIndex()
        with pytest.raises(ValueError, match="non-empty"):
            idx._build(_entries(("", ["a"])))

    def test_non_list_tokens_raise(self) -> None:
        idx = InvertedIndex()
        with pytest.raises(TypeError):
            idx._build([{"id": "x", "tokens": "single string"}])  # type: ignore

    def test_non_string_token_raises(self) -> None:
        idx = InvertedIndex()
        with pytest.raises(TypeError):
            idx._build([{"id": "x", "tokens": ["ok", 123]}])  # type: ignore


class TestRetrieval:
    def test_get_matching_doc_ids_returns_union(self, index_with_corpus) -> None:
        ids = index_with_corpus._get_matching_doc_ids(["bm25", "rag"])
        assert ids == {"d1", "d2"}

    def test_get_matching_doc_ids_unknown_token(self, index_with_corpus) -> None:
        assert index_with_corpus._get_matching_doc_ids(["unknown"]) == set()

    def test_vocabulary_contains_all_tokens(self, index_with_corpus) -> None:
        vocab = index_with_corpus._get_vocabulary()
        assert "rag" in vocab
        assert "bm25" in vocab


class TestPersistence:
    def test_round_trip(self, tmp_path: Path, index_with_corpus: InvertedIndex) -> None:
        index_with_corpus._persist(tmp_path)
        new_idx = InvertedIndex()
        new_idx._load(tmp_path)
        assert new_idx._get_tf("is", "d1") == 1
        assert new_idx._get_stats()["doc_count"] == 3

    def test_load_missing_dir_is_noop(self, tmp_path: Path) -> None:
        idx = InvertedIndex()
        idx._load(tmp_path / "missing")
        assert idx._get_stats()["doc_count"] == 0


class TestView:
    def test_view_exposes_read_only_methods(self, index_with_corpus) -> None:
        view = InvertedIndexView(index_with_corpus)
        assert view.get_tf("is", "d1") == 1
        assert view.get_stats()["doc_count"] == 3
        assert "d1" in view.get_matching_doc_ids(["rag"])
        assert "rag" in view.get_vocabulary()
