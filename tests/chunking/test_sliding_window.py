"""Tests for SlidingWindowChunker.

The chunker is purely arithmetic over character indices. Tests pin window
size, overlap, char_start/char_end metadata, and parameter validation.
"""

from __future__ import annotations

import pytest

from rag.ingestion.chunking.sliding_window_chunker import SlidingWindowChunker
from rag.ingestion.chunking.strategies import FIXED_OVERLAP


def _doc(text: str) -> dict:
    return {"id": "doc_1", "content": text, "metadata": {"source": "f.md"}}


class TestValidation:
    def test_zero_chunk_size_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            SlidingWindowChunker(0, 0, FIXED_OVERLAP)

    def test_negative_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            SlidingWindowChunker(10, -1, FIXED_OVERLAP)

    def test_overlap_geq_size_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap must be <"):
            SlidingWindowChunker(10, 10, FIXED_OVERLAP)


class TestSplitting:
    def test_short_text_yields_single_chunk(self) -> None:
        chunker = SlidingWindowChunker(100, 0, FIXED_OVERLAP)
        out = list(chunker.chunk(_doc("hello")))
        assert len(out) == 1
        assert out[0]["text"] == "hello"
        assert out[0]["metadata"]["char_start"] == 0
        assert out[0]["metadata"]["char_end"] == 5

    def test_split_with_overlap(self) -> None:
        chunker = SlidingWindowChunker(chunk_size=5, overlap=2,
                                       strategy=FIXED_OVERLAP)
        text = "abcdefghij"
        out = list(chunker.chunk(_doc(text)))
        # step = 3 → windows start at 0, 3, 6, 9
        starts = [c["metadata"]["char_start"] for c in out]
        assert starts == [0, 3, 6, 9]
        assert out[0]["text"] == "abcde"
        assert out[1]["text"] == "defgh"
        assert out[3]["text"] == "j"

    def test_no_overlap_partitions_exactly(self) -> None:
        chunker = SlidingWindowChunker(chunk_size=3, overlap=0,
                                       strategy=FIXED_OVERLAP)
        out = list(chunker.chunk(_doc("abcdefg")))
        assert [c["text"] for c in out] == ["abc", "def", "g"]

    def test_chunk_index_increments(self) -> None:
        chunker = SlidingWindowChunker(2, 0, FIXED_OVERLAP)
        out = list(chunker.chunk(_doc("abcdef")))
        assert [c["metadata"]["chunk_index"] for c in out] == [0, 1, 2]

    def test_metadata_carried_over(self) -> None:
        chunker = SlidingWindowChunker(10, 0, FIXED_OVERLAP)
        doc = {"id": "doc_1", "content": "alpha", "metadata": {"k": "v", "source": "s"}}
        out = list(chunker.chunk(doc))
        assert out[0]["metadata"]["k"] == "v"
        assert out[0]["metadata"]["source"] == "s"

    def test_char_end_within_text_bounds(self) -> None:
        chunker = SlidingWindowChunker(5, 0, FIXED_OVERLAP)
        out = list(chunker.chunk(_doc("abcdefgh")))
        for c in out:
            assert c["metadata"]["char_end"] <= 8

    def test_chunk_ids_unique_within_document(self) -> None:
        chunker = SlidingWindowChunker(2, 0, FIXED_OVERLAP)
        out = list(chunker.chunk(_doc("abcdefgh")))
        ids = [c["id"] for c in out]
        assert len(ids) == len(set(ids))

    def test_deterministic_output(self) -> None:
        chunker = SlidingWindowChunker(3, 1, FIXED_OVERLAP)
        a = list(chunker.chunk(_doc("alphabet")))
        b = list(chunker.chunk(_doc("alphabet")))
        assert a == b

    def test_empty_text_yields_no_chunks(self) -> None:
        chunker = SlidingWindowChunker(10, 0, FIXED_OVERLAP)
        out = list(chunker.chunk(_doc("")))
        assert out == []
