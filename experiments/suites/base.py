"""Suite base class.

Every experimental suite subclasses :class:`Suite` and implements ``run``.
The base class exposes per-suite output directories and a shared context
object containing all corpora, queries, generator, and run_id.

Suites are intentionally narrow: each answers exactly one research question.

After ``execute`` finishes, the full summary dict is persisted to
``outputs/aggregated/<run_id>/<suite_key>/_suite_summary.json`` so the
standalone ``report_builder.py`` can rebuild the Markdown report from disk
without re-running any experiments.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiments.adapters.generation_adapter import BuiltGenerator
from experiments.configs.settings import SETTINGS, output_subdir
from experiments.core.types import SyntheticQuery

logger = logging.getLogger(__name__)


@dataclass
class ExperimentContext:
    """Shared state passed to every suite."""

    run_id: str
    corpora_chunks: Dict[str, List[Dict]]            # corpus_name -> chunks
    queries: Dict[str, List[SyntheticQuery]]         # corpus_name -> queries
    generator: BuiltGenerator
    metadata: Dict[str, Any] = field(default_factory=dict)


def _json_safe(obj: Any) -> Any:
    if is_dataclass(obj):
        return _json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class Suite(ABC):
    """Abstract base for experiment suites.

    Subclasses populate ``key`` and ``description`` and implement ``run``.
    """

    key: str = "abstract"
    description: str = "Abstract suite"

    def __init__(self, ctx: ExperimentContext) -> None:
        self.ctx = ctx
        self.logger = logging.getLogger(f"experiments.suites.{self.key}")
        self._t0: Optional[float] = None

    # ── Output paths ────────────────────────────────────────────────────────
    @property
    def raw_dir(self) -> Path:
        return output_subdir("raw", self.ctx.run_id, self.key)

    @property
    def agg_dir(self) -> Path:
        return output_subdir("aggregated", self.ctx.run_id, self.key)

    @property
    def figures_dir(self) -> Path:
        return output_subdir("figures", self.ctx.run_id, self.key)

    @property
    def logs_dir(self) -> Path:
        return output_subdir("logs", self.ctx.run_id)

    def raw_path(self, name: str) -> Path:
        return self.raw_dir / name

    def agg_path(self, name: str) -> Path:
        return self.agg_dir / name

    def figure_path(self, name: str) -> Path:
        return self.figures_dir / name

    # ── Lifecycle ──────────────────────────────────────────────────────────
    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the suite. Return a small summary dict for the report.

        The summary dict must include at least:

            {"figures": [paths...], "tables": [paths...], "findings": [strings]}
        """

    def execute(self) -> Dict[str, Any]:
        """Wrapper that logs duration and tags errors per suite."""
        self.logger.info("Starting suite %s (%s)", self.key, self.description)
        self._t0 = time.perf_counter()
        try:
            summary = self.run()
        except Exception as exc:
            elapsed = (time.perf_counter() - (self._t0 or 0.0))
            self.logger.exception(
                "Suite %s failed after %.2fs: %s", self.key, elapsed, exc
            )
            summary = {
                "key": self.key,
                "description": self.description,
                "status": "failed",
                "duration_s": elapsed,
                "error": str(exc),
                "figures": [],
                "tables": [],
                "findings": [f"Suite {self.key} failed: {exc}"],
            }
            self._persist_summary(summary)
            return summary
        elapsed = (time.perf_counter() - (self._t0 or 0.0))
        summary.setdefault("key", self.key)
        summary.setdefault("description", self.description)
        summary.setdefault("status", "ok")
        summary["duration_s"] = elapsed
        self.logger.info("Suite %s finished in %.2fs", self.key, elapsed)
        self._persist_summary(summary)
        return summary

    # ── Persistence ──────────────────────────────────────────────────────────
    def _persist_summary(self, summary: Dict[str, Any]) -> None:
        """Write the suite summary JSON so reports can be rebuilt offline."""
        try:
            path = self.agg_dir / "_suite_summary.json"
            path.write_text(
                json.dumps(_json_safe(summary), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            self.logger.warning("Failed to persist suite summary: %s", exc)
