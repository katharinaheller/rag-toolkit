"""Tests for JSONLStore.

JSONLStore is responsible for durable, deduplicated, append-only persistence
of arbitrary records. These tests verify crash recovery (corrupt lines and
duplicates removed), write-time validation, the fsync code path, and the
RepairStats summary contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, FrozenSet, List

import pytest

from rag.ingestion.storage.dedup import (
    InMemoryDedupStrategy,
    NoDedupStrategy,
    PersistentDedupStrategy,
)
from rag.ingestion.storage.jsonl_store import JSONLStore, RepairStats
from tests.utils.helpers import read_jsonl


def _write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


class TestBasicWrites:
    def test_writes_valid_records(self, tmp_path: Path) -> None:
        store = JSONLStore(tmp_path / "out.jsonl")
        store.write_many([{"id": "1", "v": "a"}, {"id": "2", "v": "b"}])
        records = read_jsonl(tmp_path / "out.jsonl")
        assert [r["id"] for r in records] == ["1", "2"]

    def test_appends_across_calls(self, tmp_path: Path) -> None:
        store = JSONLStore(tmp_path / "out.jsonl")
        store.write_many([{"id": "1"}])
        store.write_many([{"id": "2"}])
        records = read_jsonl(tmp_path / "out.jsonl")
        assert [r["id"] for r in records] == ["1", "2"]

    def test_records_without_id_always_written(self, tmp_path: Path) -> None:
        store = JSONLStore(tmp_path / "out.jsonl")
        store.write_many([{"k": "v1"}, {"k": "v2"}, {"k": "v1"}])
        records = read_jsonl(tmp_path / "out.jsonl")
        assert len(records) == 3


class TestRequiredKeys:
    def test_missing_keys_raise(self, tmp_path: Path) -> None:
        store = JSONLStore(
            tmp_path / "out.jsonl",
            required_keys=frozenset({"id", "text"}),
        )
        with pytest.raises(ValueError, match="missing required keys"):
            store.write_many([{"id": "1"}])

    def test_complete_keys_succeed(self, tmp_path: Path) -> None:
        store = JSONLStore(
            tmp_path / "out.jsonl",
            required_keys=frozenset({"id", "text"}),
        )
        store.write_many([{"id": "1", "text": "x"}])
        assert read_jsonl(tmp_path / "out.jsonl") == [{"id": "1", "text": "x"}]


class TestDeduplication:
    def test_in_memory_dedup(self, tmp_path: Path) -> None:
        store = JSONLStore(tmp_path / "out.jsonl", dedup=InMemoryDedupStrategy())
        store.write_many([{"id": "a"}, {"id": "a"}, {"id": "b"}])
        records = read_jsonl(tmp_path / "out.jsonl")
        assert [r["id"] for r in records] == ["a", "b"]

    def test_no_dedup_strategy_allows_duplicates(self, tmp_path: Path) -> None:
        store = JSONLStore(tmp_path / "out.jsonl", dedup=NoDedupStrategy())
        store.write_many([{"id": "a"}, {"id": "a"}, {"id": "a"}])
        records = read_jsonl(tmp_path / "out.jsonl")
        assert len(records) == 3

    def test_persistent_dedup_survives_reopen(self, tmp_path: Path) -> None:
        idx_path = tmp_path / "dedup.idx"
        out_path = tmp_path / "out.jsonl"
        store_a = JSONLStore(out_path, dedup=PersistentDedupStrategy(idx_path))
        store_a.write_many([{"id": "a"}])

        store_b = JSONLStore(out_path, dedup=PersistentDedupStrategy(idx_path))
        store_b.write_many([{"id": "a"}, {"id": "b"}])

        records = read_jsonl(out_path)
        assert [r["id"] for r in records] == ["a", "b"]


class TestRepair:
    def test_drops_corrupt_lines_on_open(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_lines(out, [
            json.dumps({"id": "1", "v": "a"}),
            "{ not valid json",
            json.dumps({"id": "2", "v": "b"}),
            "another bad line",
        ])
        JSONLStore(out)
        recovered = read_jsonl(out)
        assert [r["id"] for r in recovered] == ["1", "2"]

    def test_drops_duplicate_ids_on_open(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_lines(out, [
            json.dumps({"id": "1"}),
            json.dumps({"id": "1"}),
            json.dumps({"id": "2"}),
            json.dumps({"id": "2"}),
        ])
        JSONLStore(out)
        recovered = read_jsonl(out)
        assert [r["id"] for r in recovered] == ["1", "2"]

    def test_repair_seeds_dedup_index(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_lines(out, [json.dumps({"id": "existing"})])
        store = JSONLStore(out)
        store.write_many([{"id": "existing"}, {"id": "new"}])
        recovered = read_jsonl(out)
        assert [r["id"] for r in recovered] == ["existing", "new"]

    def test_repair_on_open_disabled(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        _write_lines(out, [
            json.dumps({"id": "1"}),
            "garbage",
        ])
        JSONLStore(out, repair_on_open=False)
        assert "garbage" in out.read_text(encoding="utf-8")


class TestRepairStats:
    def test_was_clean_false_when_issues_present(self) -> None:
        s = RepairStats(total_lines=4, valid_lines=2, corrupt_lines=1,
                        duplicate_lines=1)
        assert s.was_clean is False

    def test_was_clean_true_when_no_issues(self) -> None:
        s = RepairStats(total_lines=2, valid_lines=2, corrupt_lines=0,
                        duplicate_lines=0)
        assert s.was_clean is True


class TestFsync:
    def test_fsync_path_runs_without_error(self, tmp_path: Path) -> None:
        store = JSONLStore(tmp_path / "out.jsonl", fsync=True)
        store.write_many([{"id": "1"}])
        assert read_jsonl(tmp_path / "out.jsonl") == [{"id": "1"}]
