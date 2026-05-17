"""Tests for IngestionMetrics.

Covers in-place counter mutation, the skip_sink bound-callable contract, and
the dictionary/log serialisation paths.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from rag.ingestion.metrics import IngestionMetrics


class CapturingLogger:
    """Captures structlog-style kwargs from a single .info() call."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def info(self, event: str, **kwargs: Any) -> None:
        self.events.append({"event": event, **kwargs})


class TestDefaults:
    def test_defaults_zero(self) -> None:
        m = IngestionMetrics()
        assert (m.docs_loaded, m.docs_dropped, m.chunks_produced,
                m.files_skipped, m.records_skipped) == (0, 0, 0, 0, 0)


class TestSkipSink:
    def test_sink_increments_records_skipped(self) -> None:
        m = IngestionMetrics()
        sink = m.skip_sink()
        sink()
        sink()
        sink()
        assert m.records_skipped == 3

    def test_sink_isolated_per_metrics_instance(self) -> None:
        a = IngestionMetrics()
        b = IngestionMetrics()
        a.skip_sink()()
        assert a.records_skipped == 1
        assert b.records_skipped == 0


class TestSerialisation:
    def test_as_dict_round_trip(self) -> None:
        m = IngestionMetrics(docs_loaded=2, docs_dropped=1, chunks_produced=10)
        d = m.as_dict()
        assert d == {
            "docs_loaded": 2,
            "docs_dropped": 1,
            "chunks_produced": 10,
            "files_skipped": 0,
            "records_skipped": 0,
        }

    def test_log_emits_all_counters(self) -> None:
        m = IngestionMetrics(docs_loaded=5, chunks_produced=12, records_skipped=3)
        logger = CapturingLogger()
        m.log(logger)
        assert len(logger.events) == 1
        evt = logger.events[0]
        assert evt["event"] == "rag.ingestion.metrics"
        assert evt["docs_loaded"] == 5
        assert evt["chunks_produced"] == 12
        assert evt["records_skipped"] == 3
