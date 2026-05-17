import json
from pathlib import Path
from typing import Iterable, Iterator, List

from rag.embedding.types import EmbeddingVector


class EmbeddingStore:
    """Append-only JSONL persistence for EmbeddingVector objects."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        """Truncate the store. Call once before a new orchestrator run."""
        self.path.write_text("", encoding="utf-8")

    def write_many(self, items: Iterable[EmbeddingVector]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def stream_all(self) -> Iterator[EmbeddingVector]:
        """Stream vectors one at a time from disk."""
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def read_all(self) -> List[EmbeddingVector]:
        """Materialize all stored vectors. For small datasets or inspection only."""
        return list(self.stream_all())
