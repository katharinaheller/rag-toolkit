from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class DedupStrategy(Protocol):
    """Contract for deduplication within a store."""

    def is_known(self, item_id: str) -> bool:
        ...

    def mark_seen(self, item_id: str) -> None:
        ...

    def seed(self, ids: Iterable[str]) -> None:
        ...

    def rebuild_from(self, ids: Iterable[str]) -> None:
        ...

    def reset(self) -> None:
        ...


class NoDedupStrategy:
    """Unconditional write — no deduplication."""

    def is_known(self, item_id: str) -> bool:
        return False

    def mark_seen(self, item_id: str) -> None:
        pass

    def seed(self, ids: Iterable[str]) -> None:
        pass

    def rebuild_from(self, ids: Iterable[str]) -> None:
        pass

    def reset(self) -> None:
        pass


class InMemoryDedupStrategy:
    """In-memory hash-set deduplication. Seeded from disk on open; lost on restart."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_known(self, item_id: str) -> bool:
        return item_id in self._seen

    def mark_seen(self, item_id: str) -> None:
        self._seen.add(item_id)

    def seed(self, ids: Iterable[str]) -> None:
        self._seen.update(ids)

    def rebuild_from(self, ids: Iterable[str]) -> None:
        self._seen = set(ids)

    def reset(self) -> None:
        self._seen.clear()


class PersistentDedupStrategy:
    """File-backed ID index that survives process restarts.

    IDs are appended immediately on mark_seen(). The full index is loaded into
    a hash set on __init__ for O(1) lookups.
    """

    def __init__(self, index_path: Path) -> None:
        self._path = index_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = self._load()

    def _load(self) -> set[str]:
        if not self._path.exists():
            return set()
        with self._path.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    def is_known(self, item_id: str) -> bool:
        return item_id in self._seen

    def mark_seen(self, item_id: str) -> None:
        if item_id not in self._seen:
            self._seen.add(item_id)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(item_id + "\n")

    def seed(self, ids: Iterable[str]) -> None:
        pass  # The index loads from its own file.

    def rebuild_from(self, ids: Iterable[str]) -> None:
        new_ids = set(ids)
        with self._path.open("w", encoding="utf-8") as f:
            for item_id in sorted(new_ids):
                f.write(item_id + "\n")
        self._seen = new_ids

    def reset(self) -> None:
        self._path.write_text("", encoding="utf-8")
        self._seen.clear()
