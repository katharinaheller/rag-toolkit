"""Tests for ChunkingStrategy and the named constants."""

from __future__ import annotations

import pytest

from rag.ingestion.chunking.strategies import (
    FIXED_OVERLAP,
    SCHEMA_AWARE,
    STRUCTURE_AWARE,
    SYNTAX_AWARE,
    ChunkingStrategy,
)
from rag.ingestion.ids import content_chunk_id, positional_chunk_id


class TestNames:
    @pytest.mark.parametrize("strat,name", [
        (FIXED_OVERLAP, "fixed_overlap"),
        (STRUCTURE_AWARE, "structure_aware"),
        (SYNTAX_AWARE, "syntax_aware"),
        (SCHEMA_AWARE, "schema_aware"),
    ])
    def test_named_constants(self, strat: ChunkingStrategy, name: str) -> None:
        assert strat.name == name


class TestDefaults:
    def test_default_chunk_id_fn_is_positional(self) -> None:
        assert FIXED_OVERLAP.chunk_id_fn is positional_chunk_id

    def test_custom_strategy(self) -> None:
        custom = ChunkingStrategy(name="custom", chunk_id_fn=content_chunk_id)
        assert custom.name == "custom"
        assert custom.chunk_id_fn is content_chunk_id


class TestImmutable:
    def test_strategy_is_frozen_dataclass(self) -> None:
        with pytest.raises(Exception):
            FIXED_OVERLAP.name = "different"  # type: ignore[misc]
