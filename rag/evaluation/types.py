"""Typed, immutable data models for the RAG evaluation layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvaluationExample:
    """A single labelled example used for evaluation."""

    query: str
    expected_answer: str
    relevant_document_ids: List[str]
    relevant_chunk_ids: Optional[List[str]] = None
    example_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("EvaluationExample.query must be a non-empty string.")
        if not isinstance(self.relevant_document_ids, list):
            raise TypeError("relevant_document_ids must be a list.")


@dataclass(frozen=True)
class RetrievedContext:
    """A single retrieved context as presented to the generation stage."""

    document_id: str
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedAnswer:
    """The output of the generation stage for a single query."""

    text: str
    model: str
    strategy: str
    latency_ms: float
    prompt_chars: int
    context_chars: int
    error: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def success(self) -> bool:
        """True when generation completed without error and produced a non-empty answer."""
        return self.error is None and bool(self.text)


@dataclass
class StageTiming:
    """Per-stage wall-clock timings for one prediction (milliseconds; None = not measured)."""

    retrieval_ms: Optional[float] = None
    reranking_ms: Optional[float] = None
    prompt_construction_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    end_to_end_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_ms": self.retrieval_ms,
            "reranking_ms": self.reranking_ms,
            "prompt_construction_ms": self.prompt_construction_ms,
            "generation_ms": self.generation_ms,
            "end_to_end_ms": self.end_to_end_ms,
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    """Point-in-time capture of system resource usage. Unavailable metrics are None."""

    timestamp_iso: str
    cpu_percent: Optional[float]
    memory_rss_mb: Optional[float]
    memory_peak_mb: Optional[float]
    gpu_utilisation_percent: Optional[float]
    gpu_memory_used_mb: Optional[float]
    gpu_memory_total_mb: Optional[float]
    process_id: int
    hostname: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_iso": self.timestamp_iso,
            "cpu_percent": self.cpu_percent,
            "memory_rss_mb": self.memory_rss_mb,
            "memory_peak_mb": self.memory_peak_mb,
            "gpu_utilisation_percent": self.gpu_utilisation_percent,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "process_id": self.process_id,
            "hostname": self.hostname,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationPrediction:
    """Full output of the RAG pipeline for one EvaluationExample."""

    example: EvaluationExample
    retrieved_contexts: List[RetrievedContext]
    generated_answer: Optional[GeneratedAnswer] = None
    timings: StageTiming = field(default_factory=StageTiming)
    resource_snapshots: List[ResourceSnapshot] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_retrieval(self) -> bool:
        return bool(self.retrieved_contexts)

    @property
    def has_generation(self) -> bool:
        return self.generated_answer is not None


@dataclass(frozen=True)
class MetricResult:
    """Output of a single metric computed over one or more predictions."""

    metric_name: str
    value: float
    per_example: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "per_example": self.per_example,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationRunResult:
    """Complete output of a single evaluation run over a dataset."""

    run_id: str
    config_dict: Dict[str, Any]
    predictions: List[EvaluationPrediction]
    metric_results: Dict[str, MetricResult]
    timestamp_iso: str
    duration_s: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_examples(self) -> int:
        return len(self.predictions)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config": self.config_dict,
            "n_examples": self.n_examples,
            "timestamp_iso": self.timestamp_iso,
            "duration_s": self.duration_s,
            "metrics": {k: v.to_dict() for k, v in self.metric_results.items()},
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class BenchmarkStats:
    """Aggregate statistics for a series of latency or throughput measurements."""

    n: int
    mean: float
    median: float
    min: float
    max: float
    p95: float
    std_dev: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n": self.n,
            "mean": self.mean,
            "median": self.median,
            "min": self.min,
            "max": self.max,
            "p95": self.p95,
            "std_dev": self.std_dev,
        }


@dataclass
class BenchmarkResult:
    """Output of a single benchmark run."""

    benchmark_name: str
    config_dict: Dict[str, Any]
    stats: BenchmarkStats
    raw_values: List[float]
    timestamp_iso: str
    duration_s: float
    warmup_n: int
    measured_n: int
    corpus_size: Optional[int] = None
    top_k: Optional[int] = None
    concurrency: Optional[int] = None
    model_name: Optional[str] = None
    hardware_metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark_name": self.benchmark_name,
            "config": self.config_dict,
            "stats": self.stats.to_dict(),
            "timestamp_iso": self.timestamp_iso,
            "duration_s": self.duration_s,
            "warmup_n": self.warmup_n,
            "measured_n": self.measured_n,
            "corpus_size": self.corpus_size,
            "top_k": self.top_k,
            "concurrency": self.concurrency,
            "model_name": self.model_name,
            "hardware_metadata": self.hardware_metadata,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }
