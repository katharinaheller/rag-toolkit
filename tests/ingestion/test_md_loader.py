"""Tests for MdLoader.

Covers binary detection, encoding fallback, and metadata shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from rag.ingestion.loaders.md_loader import MdLoader


@pytest.fixture
def loader() -> MdLoader:
    return MdLoader()


class TestSupports:
    def test_supports_md(self, loader: MdLoader, tmp_path: Path) -> None:
        assert loader.supports(tmp_path / "a.md") is True
        assert loader.supports(tmp_path / "a.MD") is True

    def test_rejects_others(self, loader: MdLoader, tmp_path: Path) -> None:
        assert loader.supports(tmp_path / "a.txt") is False
        assert loader.supports(tmp_path / "a.markdown") is False


class TestLoad:
    def test_loads_text_markdown(self, loader: MdLoader, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("# title\n\nHello world.", encoding="utf-8")
        docs = list(loader.load(p))
        assert len(docs) == 1
        assert docs[0]["content"].startswith("# title")
        assert docs[0]["metadata"] == {
            "source": str(p), "type": "md", "format": "markdown",
        }

    def test_skips_binary_files(self, loader: MdLoader, tmp_path: Path) -> None:
        p = tmp_path / "binary.md"
        p.write_bytes(b"PNG\x00\x01\x02\x03binary content")
        skipped: List[int] = []
        docs = list(loader.load(p, skip_sink=lambda: skipped.append(1)))
        assert docs == []
        assert skipped == [1]

    def test_latin1_fallback(self, loader: MdLoader, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_bytes(b"caf\xe9 r\xe9sum\xe9")
        docs = list(loader.load(p))
        assert len(docs) == 1
        assert "caf" in docs[0]["content"]

    def test_id_is_blank_by_default(self, loader: MdLoader, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("text", encoding="utf-8")
        docs = list(loader.load(p))
        assert docs[0]["id"] == ""
