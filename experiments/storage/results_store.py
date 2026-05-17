"""Result persistence: JSONL append-only writers + CSV exporters + manifest.

Every experiment writes per-record JSONL rows under ``outputs/raw/<run_id>/``
and an aggregated row per metric under ``outputs/aggregated/<run_id>/``. A
``manifest.json`` per run captures environment metadata.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import socket
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _serialise(obj: Any) -> Any:
    if is_dataclass(obj):
        return _serialise(asdict(obj))
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


class JsonlWriter:
    """Append-only JSONL writer. Safe for streaming many records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")

    def write(self, record: Any) -> None:
        self._fp.write(json.dumps(_serialise(record), ensure_ascii=False) + "\n")

    def write_many(self, records: Iterable[Any]) -> None:
        for r in records:
            self.write(r)

    def flush(self) -> None:
        self._fp.flush()

    def close(self) -> None:
        try:
            self._fp.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write a CSV file. ``rows`` must be a list of dicts with identical keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(_serialise(value), ensure_ascii=False)
    return value


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line in %s", path)
    return rows


def write_manifest(
    path: Path,
    run_id: str,
    settings_dict: Dict[str, Any],
    suites_planned: List[str],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Capture environment metadata for reproducibility."""
    info: Dict[str, Any] = {
        "run_id": run_id,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "settings": settings_dict,
        "suites_planned": suites_planned,
    }
    if extra:
        info.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_serialise(info), indent=2, ensure_ascii=False), encoding="utf-8")
