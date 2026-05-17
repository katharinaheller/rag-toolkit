"""Tests for DocumentStructureChunker.

Validates structure-aware splitting on Markdown AST blocks, and the fallback
contract: when parsing fails or returns no blocks, fallback chunks must be
emitted with metadata['fallback']=True.
"""

from __future__ import annotations

from typing import Iterable

import pytest

from rag.ingestion.chunking.base import Chunker
from rag.ingestion.chunking.document_structure_chunker import DocumentStructureChunker
from rag.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from rag.ingestion.chunking.strategies import FIXED_OVERLAP, STRUCTURE_AWARE
from rag.ingestion.schema import Chunk, Document


pytestmark = pytest.mark.requires_marko


def _doc(text: str) -> Document:
    return {"id": "doc_1", "content": text, "metadata": {"source": "f.md"}}


@pytest.fixture
def fallback() -> Chunker:
    return SlidingWindowChunker(chunk_size=100, overlap=0, strategy=FIXED_OVERLAP)


class TestStructureAware:
    def test_renders_headings_and_paragraphs(self, fallback: Chunker) -> None:
        chunker = DocumentStructureChunker(
            max_chunk_size=1000, fallback_chunker=fallback,
            strategy=STRUCTURE_AWARE,
        )
        text = "# Heading\n\nFirst paragraph.\n\nSecond paragraph."
        out = list(chunker.chunk(_doc(text)))
        assert len(out) >= 1
        joined = "\n\n".join(c["text"] for c in out)
        assert "Heading" in joined
        assert "First paragraph" in joined
        assert all(c["metadata"].get("strategy") == "structure_aware" for c in out)

    def test_groups_blocks_into_size_bounded_chunks(self, fallback: Chunker) -> None:
        chunker = DocumentStructureChunker(
            max_chunk_size=20, fallback_chunker=fallback, strategy=STRUCTURE_AWARE,
        )
        text = "# A\n\nfirst block.\n\n# B\n\nsecond.\n\n# C\n\nthird."
        out = list(chunker.chunk(_doc(text)))
        assert len(out) > 1

    def test_chunk_indices_sequential(self, fallback: Chunker) -> None:
        chunker = DocumentStructureChunker(
            max_chunk_size=50, fallback_chunker=fallback, strategy=STRUCTURE_AWARE,
        )
        text = "# A\n\npara.\n\n# B\n\npara.\n\n# C\n\npara."
        out = list(chunker.chunk(_doc(text)))
        indices = [c["metadata"]["chunk_index"] for c in out]
        assert indices == list(range(len(out)))


class TestFallback:
    def test_empty_after_parse_falls_back(self, fallback: Chunker) -> None:
        chunker = DocumentStructureChunker(
            max_chunk_size=100, fallback_chunker=fallback, strategy=STRUCTURE_AWARE,
        )
        # Single whitespace string still yields no rendered blocks.
        out = list(chunker.chunk(_doc(" ")))
        assert out == [] or all(c["metadata"].get("fallback") is True for c in out)

    def test_fallback_flag_set_when_parse_fails(self, fallback: Chunker,
                                                 monkeypatch) -> None:
        """If marko.parse raises, the chunker must fall back gracefully."""
        import rag.ingestion.chunking.document_structure_chunker as mod

        def boom(_text: str):
            raise RuntimeError("synthetic parse failure")

        monkeypatch.setattr(mod, "marko", type("F", (), {"parse": staticmethod(boom)}))
        chunker = DocumentStructureChunker(
            max_chunk_size=100, fallback_chunker=fallback, strategy=STRUCTURE_AWARE,
        )
        out = list(chunker.chunk(_doc("alpha beta")))
        assert all(c["metadata"].get("fallback") is True for c in out)
        assert all(c["metadata"].get("strategy") == "structure_aware" for c in out)


class TestDeterminism:
    def test_repeated_chunking_is_stable(self, fallback: Chunker) -> None:
        chunker = DocumentStructureChunker(
            max_chunk_size=200, fallback_chunker=fallback, strategy=STRUCTURE_AWARE,
        )
        text = "# Title\n\nFirst.\n\nSecond.\n\n```python\nprint(1)\n```"
        a = list(chunker.chunk(_doc(text)))
        b = list(chunker.chunk(_doc(text)))
        assert a == b
