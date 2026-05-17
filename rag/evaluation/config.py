"""Configuration for the RAG evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


_VALID_EVAL_MODES = frozenset({"retrieval", "generation", "end_to_end"})
_VALID_EXPORT_FORMATS = frozenset({"jsonl", "csv"})

_DEFAULT_METRICS_RETRIEVAL = ["context_precision", "context_recall", "mrr", "ndcg"]
_DEFAULT_METRICS_GENERATION = ["exact_match", "token_f1"]
_DEFAULT_METRICS_PERFORMANCE = ["latency", "throughput", "memory_usage"]


@dataclass(frozen=True)
class EvaluationConfig:
    """Top-level configuration for one evaluation run."""

    mode: str = "end_to_end"
    top_k: int = 10
    retrieval_metrics: List[str] = field(default_factory=lambda: list(_DEFAULT_METRICS_RETRIEVAL))
    generation_metrics: List[str] = field(default_factory=lambda: list(_DEFAULT_METRICS_GENERATION))
    performance_metrics: List[str] = field(default_factory=lambda: list(_DEFAULT_METRICS_PERFORMANCE))
    output_dir: Optional[Path] = None
    export_jsonl: bool = True
    export_csv: bool = True
    capture_resources: bool = False
    gpu_monitoring: bool = False
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.mode not in _VALID_EVAL_MODES:
            raise ValueError(
                f"EvaluationConfig.mode must be one of {sorted(_VALID_EVAL_MODES)}, "
                f"got '{self.mode}'."
            )
        if self.top_k <= 0:
            raise ValueError(f"EvaluationConfig.top_k must be positive, got {self.top_k}.")

    @property
    def all_metrics(self) -> List[str]:
        """Flat list of all enabled metric names."""
        return self.retrieval_metrics + self.generation_metrics + self.performance_metrics
