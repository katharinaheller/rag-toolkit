"""Tests for the chunking core helper.

build_chunk must produce records with deterministic IDs that include the
strategy namespace and the canonical metadata fields (chunk_index, strategy).
"""

from __future__ import annotations

import pytest

from rag.ingestion.chunking.core import build_chunk
from rag.ingestion.chunking.strategies import (
    FIXED_OVERLAP,
    STRUCTURE_AWARE,
    ChunkingStrategy,
)
from rag.ingestion.ids import content_chunk_id, positional_chunk_id


class TestBuildChunkShape:
    def test_required_fields_present(self) -> None:
        chunk = build_chunk(
            document_id="doc_1", index=0, text="hello",
            base_metadata={"source": "x"}, extra_metadata={},
            strategy=FIXED_OVERLAP,
        )
        assert set(chunk.keys()) == {"id", "document_id", "text", "metadata"}
        assert chunk["document_id"] == "doc_1"
        assert chunk["text"] == "hello"

    def test_metadata_merges_in_order(self) -> None:
        chunk = build_chunk(
            document_id="doc_1", index=3, text="hi",
            base_metadata={"source": "f.md", "type": "md"},
            extra_metadata={"char_start": 12, "char_end": 14},
            strategy=FIXED_OVERLAP,
        )
        md = chunk["metadata"]
        assert md["source"] == "f.md"
        assert md["char_start"] == 12
        assert md["char_end"] == 14
        assert md["chunk_index"] == 3
        assert md["strategy"] == "fixed_overlap"

    def test_extra_metadata_overrides_base(self) -> None:
        chunk = build_chunk(
            document_id="d", index=0, text="x",
            base_metadata={"k": "base"},
            extra_metadata={"k": "extra"},
            strategy=FIXED_OVERLAP,
        )
        assert chunk["metadata"]["k"] == "extra"

    def test_assertion_empty_doc_id_raises(self) -> None:
        with pytest.raises(AssertionError):
            build_chunk(document_id="", index=0, text="x",
                        base_metadata={}, extra_metadata={},
                        strategy=FIXED_OVERLAP)

    def test_assertion_empty_text_raises(self) -> None:
        with pytest.raises(AssertionError):
            build_chunk(document_id="d", index=0, text="",
                        base_metadata={}, extra_metadata={},
                        strategy=FIXED_OVERLAP)


class TestIdDerivation:
    def test_default_positional_id(self) -> None:
        chunk = build_chunk(
            document_id="doc_1", index=2, text="alpha",
            base_metadata={}, extra_metadata={}, strategy=FIXED_OVERLAP,
        )
        assert chunk["id"] == positional_chunk_id("doc_1", 2, "alpha", "fixed_overlap")

    def test_strategy_namespacing(self) -> None:
        a = build_chunk("doc_1", 0, "alpha", {}, {}, FIXED_OVERLAP)
        b = build_chunk("doc_1", 0, "alpha", {}, {}, STRUCTURE_AWARE)
        assert a["id"] != b["id"]

    def test_custom_content_strategy(self) -> None:
        strat = ChunkingStrategy(name="custom_content", chunk_id_fn=content_chunk_id)
        a = build_chunk("doc_1", 0, "alpha", {}, {}, strat)
        b = build_chunk("doc_1", 0, "alpha", {}, {}, strat)
        c = build_chunk("doc_1", 0, "beta", {}, {}, strat)
        assert a["id"] == b["id"]
        assert a["id"] != c["id"]
