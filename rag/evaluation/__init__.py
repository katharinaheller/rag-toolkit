"""rag.evaluation - evaluation, monitoring, benchmarking, and reporting layer."""

from rag.evaluation.config import EvaluationConfig
from rag.evaluation.dataset import dataset_from_dicts, load_jsonl_dataset, save_jsonl_dataset
from rag.evaluation.report import BenchmarkReport, EvaluationReport
from rag.evaluation.runner import EvaluationRunner
from rag.evaluation.types import (
    BenchmarkResult,
    BenchmarkStats,
    EvaluationExample,
    EvaluationPrediction,
    EvaluationRunResult,
    GeneratedAnswer,
    MetricResult,
    ResourceSnapshot,
    RetrievedContext,
    StageTiming,
)

__all__ = [
    "BenchmarkReport",
    "BenchmarkResult",
    "BenchmarkStats",
    "EvaluationConfig",
    "EvaluationExample",
    "EvaluationPrediction",
    "EvaluationReport",
    "EvaluationRunResult",
    "EvaluationRunner",
    "GeneratedAnswer",
    "MetricResult",
    "ResourceSnapshot",
    "RetrievedContext",
    "StageTiming",
    "dataset_from_dicts",
    "load_jsonl_dataset",
    "save_jsonl_dataset",
]
