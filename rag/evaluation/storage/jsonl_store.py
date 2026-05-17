"""JSONL storage for evaluation run results. Append-only, streamable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

from rag.evaluation.types import EvaluationRunResult


def _serialise(obj: Any) -> Any:
    """Recursively convert non-JSON-serialisable objects."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialise(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return _serialise(obj.to_dict())
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses
        return _serialise(dataclasses.asdict(obj))
    return obj


class EvaluationJSONLStore:
    """Append-only JSONL store: one record per prediction plus a summary line."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_run_result(self, run_result: EvaluationRunResult) -> None:
        """Append all predictions and the summary to the store."""
        with self.path.open("a", encoding="utf-8") as f:
            for pred in run_result.predictions:
                record: Dict[str, Any] = {
                    "type": "prediction",
                    "run_id": run_result.run_id,
                    "example_id": pred.example.example_id,
                    "query": pred.example.query,
                    "expected_answer": pred.example.expected_answer,
                    "relevant_document_ids": pred.example.relevant_document_ids,
                    "retrieved_document_ids": [
                        ctx.document_id for ctx in pred.retrieved_contexts
                    ],
                    "retrieved_chunk_ids": [
                        ctx.chunk_id for ctx in pred.retrieved_contexts
                    ],
                    "retrieved_scores": [
                        ctx.score for ctx in pred.retrieved_contexts
                    ],
                    "generated_answer": (
                        pred.generated_answer.text
                        if pred.generated_answer is not None else None
                    ),
                    "generation_error": (
                        pred.generated_answer.error
                        if pred.generated_answer is not None else None
                    ),
                    "timings": pred.timings.to_dict(),
                    "errors": pred.errors,
                }
                f.write(json.dumps(_serialise(record), ensure_ascii=False) + "\n")

            summary: Dict[str, Any] = {
                "type": "run_summary",
                **run_result.to_dict(),
            }
            f.write(json.dumps(_serialise(summary), ensure_ascii=False) + "\n")

    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream all records. Yields nothing if file is absent."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass  # Skip corrupt lines.

    def read_all(self) -> List[Dict[str, Any]]:
        """Materialise all records. Avoid for large files."""
        return list(self.stream())

    def write_benchmark(self, result: Any) -> None:
        """Append a BenchmarkResult summary to the store."""
        with self.path.open("a", encoding="utf-8") as f:
            record = {"type": "benchmark", **_serialise(result.to_dict())}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
