"""Tests for the Memory Usage metric."""

import pytest

from rag.evaluation.metrics.memory_usage import MemoryUsageMetric
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, ResourceSnapshot, StageTiming,
)


def _snapshot(rss_mb: float) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp_iso="2024-01-01T00:00:00.000Z",
        cpu_percent=None,
        memory_rss_mb=rss_mb,
        memory_peak_mb=rss_mb,
        gpu_utilisation_percent=None,
        gpu_memory_used_mb=None,
        gpu_memory_total_mb=None,
        process_id=1,
        hostname="localhost",
    )


def _pred(rss_values=None):
    ex = EvaluationExample(query="Q", expected_answer="A", relevant_document_ids=[])
    snaps = [_snapshot(v) for v in (rss_values or [])]
    return EvaluationPrediction(
        example=ex, retrieved_contexts=[], generated_answer=None,
        timings=StageTiming(), resource_snapshots=snaps,
    )


class TestMemoryUsageMetric:
    def setup_method(self):
        self.metric = MemoryUsageMetric()

    def test_peak_is_primary_value(self):
        pred = _pred(rss_values=[100.0, 200.0, 150.0])
        result = self.metric.evaluate([pred])
        assert result.value == pytest.approx(200.0)

    def test_multiple_predictions(self):
        preds = [
            _pred(rss_values=[100.0, 120.0]),
            _pred(rss_values=[80.0, 300.0]),
        ]
        result = self.metric.evaluate(preds)
        assert result.value == pytest.approx(300.0)

    def test_no_snapshots(self):
        result = self.metric.evaluate([_pred()])
        assert result.value == 0.0
        assert "note" in result.metadata

    def test_empty_predictions(self):
        result = self.metric.evaluate([])
        assert result.value == 0.0

    def test_unit_in_metadata(self):
        pred = _pred(rss_values=[128.0])
        result = self.metric.evaluate([pred])
        assert result.metadata["unit"] == "MB"

    def test_none_rss_values_excluded(self):
        """Snapshots with memory_rss_mb=None should be silently excluded."""
        ex = EvaluationExample(query="Q", expected_answer="A", relevant_document_ids=[])
        snap_none = ResourceSnapshot(
            timestamp_iso="2024-01-01T00:00:00.000Z",
            cpu_percent=None, memory_rss_mb=None, memory_peak_mb=None,
            gpu_utilisation_percent=None, gpu_memory_used_mb=None,
            gpu_memory_total_mb=None, process_id=1, hostname="localhost",
        )
        pred = EvaluationPrediction(
            example=ex, retrieved_contexts=[], generated_answer=None,
            timings=StageTiming(), resource_snapshots=[snap_none],
        )
        result = self.metric.evaluate([pred])
        # No usable RSS values → 0.0 with note
        assert result.value == 0.0
        assert "note" in result.metadata
