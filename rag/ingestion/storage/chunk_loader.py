import json
from pathlib import Path
from typing import Iterable

from rag.ingestion.schema import Chunk

_REQUIRED_KEYS = frozenset({"id", "document_id", "text", "metadata"})


def load_chunks(path: Path) -> Iterable[Chunk]:
    """Stream Chunks from a JSONL file.

    Raises FileNotFoundError if the path does not exist, ValueError if any
    line is invalid JSON or missing required keys.
    """
    if not path.exists():
        raise FileNotFoundError(f"Chunk store not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in chunk store {path} at line {line_no}: {exc}"
                ) from exc

            missing = _REQUIRED_KEYS - obj.keys()
            if missing:
                raise ValueError(
                    f"Chunk at {path}:{line_no} missing required keys: {sorted(missing)}"
                )

            yield obj  # type: ignore[misc]
