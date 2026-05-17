import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable

from rag.ingestion.storage.dedup import DedupStrategy, InMemoryDedupStrategy
from rag.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RepairStats:
    """Summary of what _repair_file found and removed."""
    total_lines: int
    valid_lines: int
    corrupt_lines: int
    duplicate_lines: int

    @property
    def was_clean(self) -> bool:
        return self.corrupt_lines == 0 and self.duplicate_lines == 0


class JSONLStore:
    """Append-only JSONL store with single-pass crash recovery and pluggable dedup.

    On open the file is repaired: non-JSON lines are dropped, valid lines rewritten.
    With required_keys set, write_many() raises on missing keys.
    With fsync=True, write_many() syncs before returning.
    Single-writer only — not thread or process safe.
    """

    def __init__(
        self,
        path: Path,
        *,
        required_keys: FrozenSet[str] | None = None,
        dedup: DedupStrategy | None = None,
        repair_on_open: bool = True,
        fsync: bool = False,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._required_keys = required_keys
        self._dedup: DedupStrategy = dedup if dedup is not None else InMemoryDedupStrategy()
        self._fsync = fsync

        if repair_on_open:
            stats = self._repair_file()
            if not stats.was_clean:
                logger.warning(
                    "jsonl_store.repair",
                    path=str(self.path),
                    corrupt_lines=stats.corrupt_lines,
                    duplicate_lines=stats.duplicate_lines,
                    valid_lines=stats.valid_lines,
                )

    def _repair_file(self) -> RepairStats:
        """Repair the store in one pass. Drops corrupt and duplicate lines."""
        if not self.path.exists():
            return RepairStats(total_lines=0, valid_lines=0, corrupt_lines=0, duplicate_lines=0)

        valid: list[str] = []
        seen_ids: list[str] = []
        seen_set: set[str] = set()
        total = 0
        corrupt = 0
        duplicates = 0

        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                total += 1
                try:
                    obj = json.loads(stripped)
                    item_id = obj.get("id", "")
                    if item_id and item_id in seen_set:
                        duplicates += 1
                        continue
                    if item_id:
                        seen_set.add(item_id)
                        seen_ids.append(item_id)
                    valid.append(stripped + "\n")
                except json.JSONDecodeError:
                    corrupt += 1

        with self.path.open("w", encoding="utf-8") as f:
            f.writelines(valid)

        self._dedup.seed(seen_ids)

        return RepairStats(
            total_lines=total,
            valid_lines=len(valid),
            corrupt_lines=corrupt,
            duplicate_lines=duplicates,
        )

    def write_many(self, items: Iterable[Dict[str, Any]]) -> None:
        """Append items, respecting the deduplication strategy.

        Items without an 'id' field are always written (no dedup possible).
        """
        with self.path.open("a", encoding="utf-8") as f:
            for obj in items:
                if self._required_keys:
                    missing = self._required_keys - obj.keys()
                    if missing:
                        raise ValueError(
                            f"write_many: item missing required keys {sorted(missing)} "
                            f"(id={obj.get('id', '<no id>')})"
                        )

                item_id = obj.get("id", "")
                if item_id and self._dedup.is_known(item_id):
                    continue
                if item_id:
                    self._dedup.mark_seen(item_id)
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

            if self._fsync:
                f.flush()
                os.fsync(f.fileno())
