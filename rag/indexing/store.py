import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from rag.indexing.types import StoredDocument


class DocumentStore:
    """Append-only JSONL persistence for StoredDocument records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def reset(self) -> None:
        self.path.write_text("", encoding="utf-8")

    def write_many(self, items: Iterable[StoredDocument]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def stream_all(self) -> Iterator[StoredDocument]:
        """Stream records one at a time. Yields nothing if the store is absent."""
        if not self.path.exists():
            yield from ()
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def load_all(self) -> List[StoredDocument]:
        return list(self.stream_all())

    def load_by_id(self) -> Dict[str, StoredDocument]:
        return {doc["id"]: doc for doc in self.stream_all()}
