"""Generic helpers shared across the test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    """Write an iterable of dicts as JSONL to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    """Read all non-empty JSONL records from path."""
    if not path.exists():
        return []
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write content to a file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)


def write_bytes(path: Path, content: bytes) -> None:
    """Write raw bytes to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
