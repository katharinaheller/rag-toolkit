"""Configuration for the RAG benchmarking layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    name: str
    n_iterations: int = 30
    warmup_iterations: int = 3
    top_k: int = 10
    output_dir: Optional[Path] = None
    export_jsonl: bool = True
    capture_resources: bool = False
    gpu_monitoring: bool = False
    model_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("BenchmarkConfig.name must be a non-empty string.")
        if self.n_iterations < 1:
            raise ValueError(
                f"BenchmarkConfig.n_iterations must be >= 1, got {self.n_iterations}."
            )
        if self.warmup_iterations < 0:
            raise ValueError(
                f"BenchmarkConfig.warmup_iterations must be >= 0, got {self.warmup_iterations}."
            )
        if self.top_k <= 0:
            raise ValueError(
                f"BenchmarkConfig.top_k must be positive, got {self.top_k}."
            )


@dataclass(frozen=True)
class ScalingConfig:
    """Configuration for corpus-size or concurrency scaling experiments."""

    corpus_sizes: List[int] = field(default_factory=lambda: [100, 500, 1000, 5000])
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    top_k_values: List[int] = field(default_factory=lambda: [5, 10, 20, 50])

    def __post_init__(self) -> None:
        for size in self.corpus_sizes:
            if size <= 0:
                raise ValueError(f"corpus_sizes must be positive, got {size}.")
        for level in self.concurrency_levels:
            if level <= 0:
                raise ValueError(f"concurrency_levels must be positive, got {level}.")
        for k in self.top_k_values:
            if k <= 0:
                raise ValueError(f"top_k_values must be positive, got {k}.")
