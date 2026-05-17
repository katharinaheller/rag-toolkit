"""Tests for the private text reader helpers.

Binary detection (null-byte sniffing in first 1 KiB) and the UTF-8 → Latin-1
fallback chain together govern loader robustness; both must behave predictably
on edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.loaders._text_reader import is_binary, read_text


class TestIsBinary:
    def test_text_file_not_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_text("hello world", encoding="utf-8")
        assert is_binary(p) is False

    def test_null_byte_in_first_kib_flags_binary(self, tmp_path: Path) -> None:
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello\x00world")
        assert is_binary(p) is True

    def test_null_byte_after_first_kib_not_detected(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_bytes(b"x" * 2048 + b"\x00")
        assert is_binary(p) is False

    def test_nonexistent_path_returns_true(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.bin"
        assert is_binary(p) is True


class TestReadText:
    def test_utf8(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_text("héllo", encoding="utf-8")
        assert read_text(p) == "héllo"

    def test_latin1_fallback(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_bytes(b"caf\xe9")
        out = read_text(p)
        assert out is not None
        # Latin-1 decodes 0xE9 to é.
        assert "caf" in out

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "missing.txt"
        assert read_text(p) is None
