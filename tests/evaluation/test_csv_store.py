"""Tests for the CSV evaluation result store."""

import csv
import tempfile
from pathlib import Path

import pytest

from rag.evaluation.storage.csv_store import EvaluationCSVStore
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, EvaluationRunResult,
    GeneratedAnswer, MetricResult, StageTiming,
)


def _run(query="Q", expected="A", generated="A") -> EvaluationRunResult:
    ex = EvaluationExample(query=query, expected_answer=expected, relevant_document_ids=[])
    answer = GeneratedAnswer(
        text=generated, model="m", strategy="s",
        latency_ms=50.0, prompt_chars=10, context_chars=5,
    )
    pred = EvaluationPrediction(
        example=ex, retrieved_contexts=[],
        generated_answer=answer, timings=StageTiming(end_to_end_ms=50.0),
    )
    metric = MetricResult(metric_name="exact_match", value=1.0, per_example=[1.0])
    return EvaluationRunResult(
        run_id="csv_run",
        config_dict={"mode": "end_to_end"},
        predictions=[pred],
        metric_results={"exact_match": metric},
        timestamp_iso="2024-01-01T00:00:00.000Z",
        duration_s=1.0,
    )


class TestEvaluationCSVStore:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            store = EvaluationCSVStore(path)
            store.write_run_result(_run())
            rows = store.read_all()
            assert len(rows) == 1
            assert rows[0]["query"] == "Q"

    def test_per_example_metric_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            store = EvaluationCSVStore(path)
            store.write_run_result(_run())
            rows = store.read_all()
            assert "metric_exact_match" in rows[0]
            assert rows[0]["metric_exact_match"] == "1.0"

    def test_append_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            store = EvaluationCSVStore(path)
            store.write_run_result(_run("Q1", "A1", "A1"))
            store.write_run_result(_run("Q2", "A2", "B2"))
            rows = store.read_all()
            assert len(rows) == 2

    def test_read_nonexistent_returns_empty(self):
        path = Path("/tmp/does_not_exist_xyz.csv")
        store = EvaluationCSVStore(path)
        assert store.read_all() == []

    def test_header_written_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            store = EvaluationCSVStore(path)
            store.write_run_result(_run())
            store.write_run_result(_run())
            # Count header lines manually
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip()]
            # First line is header; subsequent are data rows
            assert lines[0].startswith("run_id")
            assert len(lines) == 3  # 1 header + 2 data rows

    def test_generated_answer_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.csv"
            store = EvaluationCSVStore(path)
            store.write_run_result(_run(generated="My answer"))
            rows = store.read_all()
            assert rows[0]["generated_answer"] == "My answer"
