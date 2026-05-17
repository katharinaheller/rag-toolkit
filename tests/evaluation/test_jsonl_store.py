"""Tests for the JSONL evaluation result store."""

import json
import tempfile
from pathlib import Path

import pytest

from rag.evaluation.storage.jsonl_store import EvaluationJSONLStore
from rag.evaluation.types import (
    EvaluationExample, EvaluationPrediction, EvaluationRunResult,
    MetricResult, StageTiming,
)


def _minimal_run_result() -> EvaluationRunResult:
    ex = EvaluationExample(query="Q", expected_answer="A", relevant_document_ids=["d1"])
    pred = EvaluationPrediction(
        example=ex, retrieved_contexts=[], generated_answer=None, timings=StageTiming()
    )
    metric = MetricResult(metric_name="exact_match", value=0.5)
    return EvaluationRunResult(
        run_id="test_run",
        config_dict={"mode": "retrieval"},
        predictions=[pred],
        metric_results={"exact_match": metric},
        timestamp_iso="2024-01-01T00:00:00.000Z",
        duration_s=1.0,
    )


class TestEvaluationJSONLStore:
    def test_write_and_stream(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            store = EvaluationJSONLStore(path)
            run = _minimal_run_result()
            store.write_run_result(run)

            records = list(store.stream())
            # One prediction record + one summary record
            assert len(records) == 2

    def test_prediction_record_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            store = EvaluationJSONLStore(path)
            store.write_run_result(_minimal_run_result())

            records = list(store.stream())
            pred_rec = next(r for r in records if r.get("type") == "prediction")
            assert pred_rec["query"] == "Q"
            assert pred_rec["expected_answer"] == "A"
            assert pred_rec["run_id"] == "test_run"

    def test_summary_record_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.jsonl"
            store = EvaluationJSONLStore(path)
            store.write_run_result(_minimal_run_result())

            records = list(store.stream())
            summary = next(r for r in records if r.get("type") == "run_summary")
            assert summary["run_id"] == "test_run"
            assert "metrics" in summary

    def test_stream_nonexistent_file(self):
        path = Path("/tmp/does_not_exist_xyz.jsonl")
        store = EvaluationJSONLStore(path)
        records = list(store.stream())
        assert records == []

    def test_append_multiple_runs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.jsonl"
            store = EvaluationJSONLStore(path)
            store.write_run_result(_minimal_run_result())
            store.write_run_result(_minimal_run_result())
            records = list(store.stream())
            # 2 runs × 2 records each = 4
            assert len(records) == 4

    def test_utf8_characters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "utf8.jsonl"
            store = EvaluationJSONLStore(path)
            ex = EvaluationExample(
                query="Qu'est-ce que le RAG?",
                expected_answer="Génération augmentée par récupération",
                relevant_document_ids=["doc_1"],
            )
            pred = EvaluationPrediction(
                example=ex, retrieved_contexts=[], generated_answer=None, timings=StageTiming()
            )
            run = EvaluationRunResult(
                run_id="utf8_run", config_dict={}, predictions=[pred],
                metric_results={}, timestamp_iso="2024-01-01T00:00:00.000Z", duration_s=0.0,
            )
            store.write_run_result(run)
            records = list(store.stream())
            pred_rec = next(r for r in records if r.get("type") == "prediction")
            assert "Génération" in pred_rec["expected_answer"]
