"""Tests for the DocumentStore JSONL persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.indexing.store import DocumentStore
from tests.fixtures.factories import make_stored_document


def test_creates_parent_directory(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "docs.jsonl"
    DocumentStore(p)
    assert p.parent.exists()


def test_write_many_appends(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "docs.jsonl")
    store.reset()
    store.write_many([make_stored_document(doc_id="a"), make_stored_document(doc_id="b")])
    rows = store.load_all()
    assert {r["id"] for r in rows} == {"a", "b"}


def test_reset_clears_file(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "docs.jsonl")
    store.write_many([make_stored_document(doc_id="x")])
    store.reset()
    assert store.load_all() == []


def test_load_by_id_keyed_correctly(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "docs.jsonl")
    store.reset()
    store.write_many([
        make_stored_document(doc_id="x", text="hello"),
        make_stored_document(doc_id="y", text="world"),
    ])
    out = store.load_by_id()
    assert out["x"]["text"] == "hello"
    assert out["y"]["text"] == "world"


def test_stream_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "d.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n\n", encoding="utf-8")
    store = DocumentStore(p)
    assert list(store.stream_all()) == []


def test_stream_missing_file_yields_empty(tmp_path: Path) -> None:
    store = DocumentStore(tmp_path / "absent.jsonl")
    assert list(store.stream_all()) == []
