"""Tests for deterministic ID derivation.

Document IDs and chunk IDs are content-addressed and namespace-prefixed.
Changes here cascade across the entire indexing pipeline, so determinism,
collision-resistance and strategy-namespacing must be verified.
"""

from __future__ import annotations

import hashlib

import pytest

from rag.ingestion.ids import (
    content_chunk_id,
    document_id_from_natural_key,
    document_id_from_text,
    positional_chunk_id,
    resolve_document_id,
)


class TestDocumentIdFromText:
    def test_prefixed_and_hex(self) -> None:
        did = document_id_from_text("hello")
        assert did.startswith("doc_")
        assert len(did) == 4 + 16

    def test_same_input_same_output(self) -> None:
        assert document_id_from_text("alpha") == document_id_from_text("alpha")

    def test_different_input_different_output(self) -> None:
        assert document_id_from_text("alpha") != document_id_from_text("beta")


class TestDocumentIdFromNaturalKey:
    def test_strip_whitespace_in_key(self) -> None:
        a = document_id_from_natural_key("  /path/to/file.md  ")
        b = document_id_from_natural_key("/path/to/file.md")
        assert a == b

    def test_different_keys_different_ids(self) -> None:
        a = document_id_from_natural_key("k1")
        b = document_id_from_natural_key("k2")
        assert a != b


class TestPositionalChunkId:
    def test_stable_under_content_changes(self) -> None:
        a = positional_chunk_id("doc1", 0, "alpha", "strategy_x")
        b = positional_chunk_id("doc1", 0, "beta", "strategy_x")
        assert a == b

    def test_varies_with_index(self) -> None:
        a = positional_chunk_id("doc1", 0, "x", "s")
        b = positional_chunk_id("doc1", 1, "x", "s")
        assert a != b

    def test_strategy_namespaces_ids(self) -> None:
        a = positional_chunk_id("doc1", 0, "x", "strat_a")
        b = positional_chunk_id("doc1", 0, "x", "strat_b")
        assert a != b


class TestContentChunkId:
    def test_changes_with_text(self) -> None:
        a = content_chunk_id("doc1", 0, "alpha", "strat")
        b = content_chunk_id("doc1", 0, "beta", "strat")
        assert a != b

    def test_stable_for_same_text(self) -> None:
        a = content_chunk_id("doc1", 0, "alpha", "strat")
        b = content_chunk_id("doc1", 0, "alpha", "strat")
        assert a == b

    def test_strategy_namespaced(self) -> None:
        a = content_chunk_id("doc1", 0, "alpha", "s1")
        b = content_chunk_id("doc1", 0, "alpha", "s2")
        assert a != b


class TestResolveDocumentId:
    def test_uses_natural_key_when_present(self) -> None:
        did = resolve_document_id("source.md", "content irrelevant")
        assert did == document_id_from_natural_key("source.md")

    def test_falls_back_to_content_hash_for_empty_key(self) -> None:
        did = resolve_document_id("", "alpha")
        assert did == document_id_from_text("alpha")

    def test_falls_back_to_content_hash_for_none(self) -> None:
        did = resolve_document_id(None, "alpha")
        assert did == document_id_from_text("alpha")

    @pytest.mark.parametrize("key", [" ", "\t", "\n", "  \t  "])
    def test_whitespace_only_key_falls_back(self, key: str) -> None:
        did = resolve_document_id(key, "alpha")
        assert did == document_id_from_text("alpha")


class TestCollisionResistance:
    def test_many_distinct_keys_yield_distinct_ids(self) -> None:
        ids = {document_id_from_natural_key(f"key_{i}") for i in range(2000)}
        assert len(ids) == 2000

    def test_chunk_ids_within_one_doc_unique(self) -> None:
        chunk_ids = {
            positional_chunk_id("doc_abc", i, "ignored", "fixed_overlap")
            for i in range(2000)
        }
        assert len(chunk_ids) == 2000
