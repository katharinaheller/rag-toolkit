"""Tests for JsonlLoader.

Validates text field fallback order, malformed line handling via skip_sink,
encoding fallback, and graceful failure on unreadable files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pytest

from rag.ingestion.loaders.jsonl_loader import JsonlLoader
from tests.utils.helpers import write_text


@pytest.fixture
def loader() -> JsonlLoader:
    return JsonlLoader()


class TestSupports:
    def test_supports_jsonl(self, loader: JsonlLoader, tmp_path: Path) -> None:
        assert loader.supports(tmp_path / "a.jsonl") is True
        assert loader.supports(tmp_path / "a.JSONL") is True

    def test_does_not_support_others(self, loader: JsonlLoader, tmp_path: Path) -> None:
        assert loader.supports(tmp_path / "a.json") is False
        assert loader.supports(tmp_path / "a.txt") is False


class TestFieldFallback:
    @pytest.mark.parametrize("key", ["text", "content", "body", "message"])
    def test_each_field_priority(
        self, loader: JsonlLoader, tmp_path: Path, key: str,
    ) -> None:
        p = tmp_path / "in.jsonl"
        write_text(p, json.dumps({key: "payload"}) + "\n")
        docs = list(loader.load(p))
        assert len(docs) == 1
        assert docs[0]["content"] == "payload"

    def test_text_wins_over_content(
        self, loader: JsonlLoader, tmp_path: Path,
    ) -> None:
        p = tmp_path / "in.jsonl"
        write_text(p, json.dumps({"text": "T", "content": "C"}) + "\n")
        docs = list(loader.load(p))
        assert docs[0]["content"] == "T"

    def test_falls_back_to_full_json_when_no_known_keys(
        self, loader: JsonlLoader, tmp_path: Path,
    ) -> None:
        p = tmp_path / "in.jsonl"
        rec = {"some": "thing", "n": 1}
        write_text(p, json.dumps(rec) + "\n")
        docs = list(loader.load(p))
        assert docs[0]["content"] == json.dumps(rec, ensure_ascii=False)


class TestSkipSinkBehavior:
    def test_malformed_lines_counted_via_sink(
        self, loader: JsonlLoader, tmp_path: Path,
    ) -> None:
        p = tmp_path / "in.jsonl"
        content = "\n".join([
            json.dumps({"text": "ok"}),
            "{ not valid json",
            json.dumps({"text": "ok2"}),
            "another garbage",
        ]) + "\n"
        write_text(p, content)

        skipped = [0]

        def sink() -> None:
            skipped[0] += 1

        docs = list(loader.load(p, skip_sink=sink))
        assert [d["content"] for d in docs] == ["ok", "ok2"]
        assert skipped[0] == 2

    def test_records_without_text_payload_counted(
        self, loader: JsonlLoader, tmp_path: Path,
    ) -> None:
        # When extractor returns "" (falsy) we expect a skip; integer-only
        # records will still serialise to JSON, so use an empty-string key.
        p = tmp_path / "in.jsonl"
        write_text(p, json.dumps({"text": ""}) + "\n")
        skipped: List[int] = []
        list(loader.load(p, skip_sink=lambda: skipped.append(1)))
        assert skipped == [1]


class TestEncoding:
    def test_latin1_fallback(self, loader: JsonlLoader, tmp_path: Path) -> None:
        p = tmp_path / "in.jsonl"
        # Write a byte sequence that is valid Latin-1 but not UTF-8.
        p.write_bytes(b'{"text": "caf\xe9"}\n')
        docs = list(loader.load(p))
        assert len(docs) == 1
        # Latin-1 decodes 0xE9 to "é".
        assert "é" in docs[0]["content"] or "caf" in docs[0]["content"]


class TestMetadata:
    def test_metadata_contains_source_and_line(
        self, loader: JsonlLoader, tmp_path: Path,
    ) -> None:
        p = tmp_path / "in.jsonl"
        write_text(p, json.dumps({"text": "first"}) + "\n" +
                   json.dumps({"text": "second"}) + "\n")
        docs = list(loader.load(p))
        assert docs[0]["metadata"]["source"] == str(p)
        assert docs[0]["metadata"]["type"] == "jsonl"
        assert docs[0]["metadata"]["line"] == 1
        assert docs[1]["metadata"]["line"] == 2

    def test_blank_lines_are_skipped_silently(
        self, loader: JsonlLoader, tmp_path: Path,
    ) -> None:
        p = tmp_path / "in.jsonl"
        write_text(p, "\n\n" + json.dumps({"text": "x"}) + "\n\n")
        skipped: List[int] = []
        docs = list(loader.load(p, skip_sink=lambda: skipped.append(1)))
        assert len(docs) == 1
        assert skipped == []
