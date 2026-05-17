"""Tests for the JSONL-backed EmbeddingStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.embedding.store import EmbeddingStore
from tests.fixtures.factories import make_embedding_vector
from tests.utils.assertions import assert_jsonl_lines


def test_init_creates_parent_directory(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "deeper" / "store.jsonl"
    EmbeddingStore(p)
    assert p.parent.exists()


def test_reset_truncates_file(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.write_text("garbage\n")
    EmbeddingStore(p).reset()
    assert p.read_text() == ""


def test_write_many_appends_records(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "s.jsonl")
    store.reset()
    batch = [make_embedding_vector(chunk_id=f"c{i}") for i in range(3)]
    store.write_many(batch)
    assert_jsonl_lines(tmp_path / "s.jsonl", 3)


def test_write_many_is_append_only(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "s.jsonl")
    store.reset()
    store.write_many([make_embedding_vector(chunk_id="a")])
    store.write_many([make_embedding_vector(chunk_id="b")])
    assert_jsonl_lines(tmp_path / "s.jsonl", 2)


def test_stream_all_yields_records_in_order(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "s.jsonl")
    store.reset()
    batch = [make_embedding_vector(chunk_id=f"c{i}") for i in range(5)]
    store.write_many(batch)
    streamed = list(store.stream_all())
    assert [r["chunk_id"] for r in streamed] == [f"c{i}" for i in range(5)]


def test_read_all_materializes_records(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "s.jsonl")
    store.reset()
    store.write_many([make_embedding_vector(chunk_id="x")])
    rows = store.read_all()
    assert len(rows) == 1 and rows[0]["chunk_id"] == "x"


def test_stream_all_handles_missing_file(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "absent.jsonl")
    assert list(store.stream_all()) == []


def test_blank_lines_in_store_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "s.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"id":"x","chunk_id":"c","document_id":"d","embedding":null,'
                 '"sparse_embedding":null,"embedding_type":"dense",'
                 '"model_type":"t","projection_method":"none","metadata":{}}\n'
                 '\n\n')
    store = EmbeddingStore(p)
    assert len(store.read_all()) == 1


def test_unicode_round_trip(tmp_path: Path) -> None:
    store = EmbeddingStore(tmp_path / "s.jsonl")
    store.reset()
    rec = make_embedding_vector(chunk_id="ünïcödé")
    rec["metadata"]["title"] = "テスト"
    store.write_many([rec])
    restored = store.read_all()[0]
    assert restored["chunk_id"] == "ünïcödé"
    assert restored["metadata"]["title"] == "テスト"
