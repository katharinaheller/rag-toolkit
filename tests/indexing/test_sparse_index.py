"""Tests for the lexical SparseIndex wrapper around InvertedIndex."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.config import SparseIndexConfig
from rag.indexing.sparse.sparse_index import SparseIndex


def _entries(*pairs):
    return [{"id": rid, "tokens": toks} for rid, toks in pairs]


@pytest.fixture
def sparse_index(tmp_path: Path) -> SparseIndex:
    idx = SparseIndex(SparseIndexConfig(), tmp_path)
    idx.add(_entries(
        ("doc_a", ["rag", "system"]),
        ("doc_b", ["bm25", "ranking"]),
        ("doc_c", ["rag", "ranking"]),
    ))
    return idx


def test_query_returns_matching_ids_sorted(sparse_index: SparseIndex) -> None:
    out = sparse_index.query(["rag"], k=10)
    ids = [r["id"] for r in out]
    assert ids == ["doc_a", "doc_c"]


def test_query_empty_for_zero_k(sparse_index: SparseIndex) -> None:
    assert sparse_index.query(["rag"], k=0) == []


def test_query_empty_for_empty_tokens(sparse_index: SparseIndex) -> None:
    assert sparse_index.query([], k=10) == []


def test_query_rejects_non_list_input(sparse_index: SparseIndex) -> None:
    with pytest.raises(TypeError):
        sparse_index.query("rag", k=1)  # type: ignore[arg-type]


def test_query_rejects_non_string_tokens(sparse_index: SparseIndex) -> None:
    with pytest.raises(TypeError):
        sparse_index.query(["valid", 99], k=1)  # type: ignore[list-item]


def test_query_returns_no_score(sparse_index: SparseIndex) -> None:
    """Sparse index intentionally returns only id; scoring belongs to retrieval."""
    out = sparse_index.query(["rag"], k=10)
    for entry in out:
        assert "score" not in entry


def test_get_view_returns_view(sparse_index: SparseIndex) -> None:
    view = sparse_index.get_view()
    assert view.get_stats()["doc_count"] == 3


def test_persist_and_load(tmp_path: Path) -> None:
    idx = SparseIndex(SparseIndexConfig(), tmp_path)
    idx.add(_entries(("d", ["one", "two"])))
    idx.persist()

    reloaded = SparseIndex(SparseIndexConfig(), tmp_path)
    reloaded.load()
    assert reloaded.get_view().get_stats()["doc_count"] == 1
