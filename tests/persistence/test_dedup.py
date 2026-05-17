"""Tests for the deduplication strategy implementations.

NoDedup, InMemoryDedup, and PersistentDedup must each conform to the
DedupStrategy protocol and behave as documented across is_known, mark_seen,
seed, rebuild_from, and reset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.storage.dedup import (
    DedupStrategy,
    InMemoryDedupStrategy,
    NoDedupStrategy,
    PersistentDedupStrategy,
)


class TestProtocol:
    def test_no_dedup_satisfies_protocol(self) -> None:
        assert isinstance(NoDedupStrategy(), DedupStrategy)

    def test_in_memory_satisfies_protocol(self) -> None:
        assert isinstance(InMemoryDedupStrategy(), DedupStrategy)

    def test_persistent_satisfies_protocol(self, tmp_path: Path) -> None:
        assert isinstance(PersistentDedupStrategy(tmp_path / "i.idx"), DedupStrategy)


class TestNoDedup:
    def test_is_known_always_false(self) -> None:
        s = NoDedupStrategy()
        s.mark_seen("a")
        assert s.is_known("a") is False

    def test_seed_and_reset_noop(self) -> None:
        s = NoDedupStrategy()
        s.seed(["a", "b"])
        s.reset()
        assert s.is_known("a") is False


class TestInMemoryDedup:
    def test_round_trip(self) -> None:
        s = InMemoryDedupStrategy()
        assert s.is_known("x") is False
        s.mark_seen("x")
        assert s.is_known("x") is True

    def test_seed(self) -> None:
        s = InMemoryDedupStrategy()
        s.seed(["a", "b"])
        assert s.is_known("a") and s.is_known("b")

    def test_rebuild_clears_and_replaces(self) -> None:
        s = InMemoryDedupStrategy()
        s.mark_seen("old")
        s.rebuild_from(["new"])
        assert s.is_known("new")
        assert s.is_known("old") is False

    def test_reset(self) -> None:
        s = InMemoryDedupStrategy()
        s.mark_seen("a")
        s.reset()
        assert s.is_known("a") is False


class TestPersistentDedup:
    def test_round_trip_persists_across_instances(self, tmp_path: Path) -> None:
        idx = tmp_path / "dedup.idx"
        s1 = PersistentDedupStrategy(idx)
        s1.mark_seen("a")
        s1.mark_seen("b")

        s2 = PersistentDedupStrategy(idx)
        assert s2.is_known("a") is True
        assert s2.is_known("b") is True

    def test_mark_seen_is_idempotent(self, tmp_path: Path) -> None:
        idx = tmp_path / "dedup.idx"
        s = PersistentDedupStrategy(idx)
        s.mark_seen("a")
        s.mark_seen("a")
        s.mark_seen("a")
        with idx.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        assert lines == ["a"]

    def test_rebuild_writes_sorted_unique_ids(self, tmp_path: Path) -> None:
        idx = tmp_path / "dedup.idx"
        s = PersistentDedupStrategy(idx)
        s.rebuild_from(["c", "a", "b", "a"])
        with idx.open("r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        assert lines == ["a", "b", "c"]

    def test_reset_truncates_file(self, tmp_path: Path) -> None:
        idx = tmp_path / "dedup.idx"
        s = PersistentDedupStrategy(idx)
        s.mark_seen("a")
        s.reset()
        assert idx.read_text(encoding="utf-8") == ""
        assert s.is_known("a") is False
