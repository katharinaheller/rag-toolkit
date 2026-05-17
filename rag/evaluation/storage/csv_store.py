"""CSV storage for evaluation run results.

One row per prediction. Per-example metric values appear as additional columns.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from rag.evaluation.types import EvaluationRunResult


_BASE_FIELDNAMES = [
    "run_id",
    "example_id",
    "query",
    "expected_answer",
    "generated_answer",
    "generation_error",
    "n_retrieved",
    "retrieved_document_ids",
    "end_to_end_ms",
    "retrieval_ms",
    "generation_ms",
    "errors",
]


class EvaluationCSVStore:
    """Append-only CSV store for evaluation results."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_run_result(self, run_result: EvaluationRunResult) -> None:
        """Append all predictions to the CSV file (header written only when new)."""
        per_example_cols: Dict[str, List[float]] = {}
        for metric_name, result in run_result.metric_results.items():
            if result.per_example is not None and len(result.per_example) == len(
                run_result.predictions
            ):
                per_example_cols[f"metric_{metric_name}"] = result.per_example

        fieldnames = _BASE_FIELDNAMES + sorted(per_example_cols.keys())
        file_exists = self.path.exists() and self.path.stat().st_size > 0

        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, extrasaction="ignore"
            )
            if not file_exists:
                writer.writeheader()

            for i, pred in enumerate(run_result.predictions):
                row: Dict[str, Any] = {
                    "run_id": run_result.run_id,
                    "example_id": pred.example.example_id or "",
                    "query": pred.example.query,
                    "expected_answer": pred.example.expected_answer,
                    "generated_answer": (
                        pred.generated_answer.text
                        if pred.generated_answer is not None else ""
                    ),
                    "generation_error": (
                        pred.generated_answer.error or ""
                        if pred.generated_answer is not None else ""
                    ),
                    "n_retrieved": len(pred.retrieved_contexts),
                    "retrieved_document_ids": json.dumps(
                        [ctx.document_id for ctx in pred.retrieved_contexts]
                    ),
                    "end_to_end_ms": pred.timings.end_to_end_ms or "",
                    "retrieval_ms": pred.timings.retrieval_ms or "",
                    "generation_ms": pred.timings.generation_ms or "",
                    "errors": "; ".join(pred.errors),
                }
                for col, values in per_example_cols.items():
                    row[col] = values[i]

                writer.writerow(row)

    def read_all(self) -> List[Dict[str, str]]:
        """Read all rows into memory. Returns empty list if file does not exist."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)
