"""Load and validate evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from rag.evaluation.types import EvaluationExample


def load_jsonl_dataset(path: Path) -> List[EvaluationExample]:
    """Load EvaluationExamples from a JSONL file.

    Required fields per line: 'query', 'expected_answer', 'relevant_document_ids'.
    Optional: 'relevant_chunk_ids', 'example_id', 'metadata'.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: List[EvaluationExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_no} in {path}: {exc}"
                ) from exc

            example = _parse_example(obj, line_no, path)
            examples.append(example)

    return examples


def _parse_example(obj: dict, line_no: int, path: Path) -> EvaluationExample:
    required = ("query", "expected_answer", "relevant_document_ids")
    for key in required:
        if key not in obj:
            raise ValueError(
                f"Dataset line {line_no} in {path} is missing required field '{key}'."
            )

    return EvaluationExample(
        query=str(obj["query"]),
        expected_answer=str(obj["expected_answer"]),
        relevant_document_ids=list(obj["relevant_document_ids"]),
        relevant_chunk_ids=(
            list(obj["relevant_chunk_ids"]) if "relevant_chunk_ids" in obj else None
        ),
        example_id=obj.get("example_id"),
        metadata=dict(obj.get("metadata", {})),
    )


def dataset_from_dicts(records: Iterable[dict]) -> List[EvaluationExample]:
    """Build an evaluation dataset directly from Python dicts (notebook-friendly)."""
    return [_parse_example(record, i + 1, Path("<inline>")) for i, record in enumerate(records)]


def save_jsonl_dataset(examples: List[EvaluationExample], path: Path) -> None:
    """Persist EvaluationExamples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            record: dict = {
                "query": ex.query,
                "expected_answer": ex.expected_answer,
                "relevant_document_ids": ex.relevant_document_ids,
            }
            if ex.relevant_chunk_ids is not None:
                record["relevant_chunk_ids"] = ex.relevant_chunk_ids
            if ex.example_id is not None:
                record["example_id"] = ex.example_id
            if ex.metadata:
                record["metadata"] = ex.metadata
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
