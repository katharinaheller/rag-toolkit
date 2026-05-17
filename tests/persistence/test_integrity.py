"""Tests for integrity verification helpers.

verify_chunk_counts and verify_chunk_ids are operational guards used at the
end of a pipeline run; they must raise informative errors so corruption is
surfaced before it propagates to embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.ingestion.storage.integrity import verify_chunk_counts, verify_chunk_ids


def _write_chunks(path: Path, ids) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for cid in ids:
            f.write(json.dumps({
                "id": cid,
                "document_id": "doc_1",
                "text": "t",
                "metadata": {},
            }) + "\n")


class TestVerifyCounts:
    def test_match_does_not_raise(self, tmp_path: Path) -> None:
        p = tmp_path / "chunks.jsonl"
        _write_chunks(p, ["c1", "c2", "c3"])
        verify_chunk_counts(3, p)

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "chunks.jsonl"
        _write_chunks(p, ["c1"])
        with pytest.raises(ValueError, match="Chunk count mismatch"):
            verify_chunk_counts(3, p)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_chunk_counts(1, tmp_path / "missing.jsonl")


class TestVerifyIds:
    def test_match_does_not_raise(self, tmp_path: Path) -> None:
        p = tmp_path / "chunks.jsonl"
        _write_chunks(p, ["c1", "c2"])
        verify_chunk_ids({"c1", "c2"}, p)

    def test_missing_ids_reported(self, tmp_path: Path) -> None:
        p = tmp_path / "chunks.jsonl"
        _write_chunks(p, ["c1"])
        with pytest.raises(ValueError, match="missing"):
            verify_chunk_ids({"c1", "c2"}, p)

    def test_unexpected_ids_reported(self, tmp_path: Path) -> None:
        p = tmp_path / "chunks.jsonl"
        _write_chunks(p, ["c1", "c_unexpected"])
        with pytest.raises(ValueError, match="unexpected"):
            verify_chunk_ids({"c1"}, p)

    def test_both_missing_and_unexpected_in_one_message(self, tmp_path: Path) -> None:
        p = tmp_path / "chunks.jsonl"
        _write_chunks(p, ["c_extra"])
        with pytest.raises(ValueError) as exc:
            verify_chunk_ids({"c_missing"}, p)
        msg = str(exc.value)
        assert "missing" in msg and "unexpected" in msg
