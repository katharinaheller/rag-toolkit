"""Suite base class.

Every experimental suite subclasses :class:`Suite` and implements ``run``.
The base class exposes per-suite output directories and a shared context
object containing all corpora, queries, generator, and run_id.

Suites are intentionally narrow: each answers exactly one research question.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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
            return {
                "key": self.key,
                "description": self.description,
                "status": "failed",
                "duration_s": elapsed,
                "error": str(exc),
                "figures": [],
                "tables": [],
                "findings": [f"Suite {self.key} failed: {exc}"],
            }
        elapsed = (time.perf_counter() - (self._t0 or 0.0))
        summary.setdefault("key", self.key)
        summary.setdefault("description", self.description)
        summary.setdefault("status", "ok")
        summary["duration_s"] = elapsed
        self.logger.info("Suite %s finished in %.2fs", self.key, elapsed)
        return summary
