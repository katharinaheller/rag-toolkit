"""Orchestrates retrieval, generation, and metric computation for one evaluation run."""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import Dict, List, Optional

from rag.evaluation.config import EvaluationConfig
from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.metrics.context_precision import ContextPrecisionMetric
from rag.evaluation.metrics.context_recall import ContextRecallMetric
from rag.evaluation.metrics.exact_match import ExactMatchMetric
from rag.evaluation.metrics.latency import LatencyMetric
from rag.evaluation.metrics.memory_usage import MemoryUsageMetric
from rag.evaluation.metrics.mrr import MRRMetric
from rag.evaluation.metrics.ndcg import NDCGMetric
from rag.evaluation.metrics.throughput import ThroughputMetric
from rag.evaluation.metrics.token_f1 import TokenF1Metric
from rag.evaluation.monitors.performance_monitor import PerformanceMonitor
from rag.evaluation.monitors.timer import Timer
from rag.evaluation.types import (
    EvaluationExample,
    EvaluationPrediction,
    EvaluationRunResult,
    GeneratedAnswer,
    MetricResult,
    RetrievedContext,
)

logger = logging.getLogger(__name__)


def _utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _build_default_metrics(
    config: EvaluationConfig, total_duration_s: Optional[float] = None
) -> List[BaseMetric]:
    """Instantiate the default metrics from config names."""
    registry: Dict[str, BaseMetric] = {
        "context_precision": ContextPrecisionMetric(),
        "context_recall": ContextRecallMetric(),
        "mrr": MRRMetric(),
        "ndcg": NDCGMetric(k=config.top_k),
        "exact_match": ExactMatchMetric(),
        "token_f1": TokenF1Metric(),
        "latency": LatencyMetric(),
        "throughput": ThroughputMetric(total_duration_s=total_duration_s),
        "memory_usage": MemoryUsageMetric(),
    }
    enabled = set(config.all_metrics)
    return [m for name, m in registry.items() if name in enabled]


class EvaluationRunner:
    """Orchestrates retrieval, generation, and metric computation.

    Supports three modes: 'retrieval' (no LLM), 'generation' (no retrieval),
    and 'end_to_end' (full pipeline). Per-example errors are captured into
    EvaluationPrediction.errors rather than raising.
    """

    def __init__(
        self,
        config: EvaluationConfig,
        retriever=None,
        strategy=None,
        metrics: Optional[List[BaseMetric]] = None,
        extra_metrics: Optional[List[BaseMetric]] = None,
    ) -> None:
        self._config = config
        self._retriever = retriever
        self._strategy = strategy
        self._custom_metrics = metrics
        self._extra_metrics = extra_metrics or []

        if config.mode in {"retrieval", "end_to_end"} and retriever is None:
            raise ValueError(
                f"EvaluationRunner: mode='{config.mode}' requires a retriever."
            )
        if config.mode in {"generation", "end_to_end"} and strategy is None:
            raise ValueError(
                f"EvaluationRunner: mode='{config.mode}' requires a strategy."
            )

    def run(self, examples: List[EvaluationExample]) -> EvaluationRunResult:
        """Execute the evaluation run over all examples."""
        run_id = self._config.run_id or str(uuid.uuid4())
        timestamp = _utc_iso()
        run_timer = Timer()
        run_timer.start()

        logger.info(
            "EvaluationRunner.run | run_id=%s | mode=%s | n=%d",
            run_id, self._config.mode, len(examples),
        )

        predictions: List[EvaluationPrediction] = []
        for ex in examples:
            pred = self._evaluate_example(ex)
            predictions.append(pred)

        total_s = (run_timer.elapsed_ms or 0.0) / 1_000.0

        # Rebuild throughput metric with the now-known total duration.
        metrics = self._build_metrics(total_s)
        metric_results: Dict[str, MetricResult] = {}
        for metric in metrics:
            try:
                result = metric.evaluate(predictions)
                metric_results[result.metric_name] = result
            except Exception as exc:
                logger.warning("Metric '%s' failed: %s", metric.name, exc)
                metric_results[metric.name] = MetricResult(
                    metric_name=metric.name,
                    value=0.0,
                    metadata={"error": str(exc)},
                )

        run_result = EvaluationRunResult(
            run_id=run_id,
            config_dict=self._config_dict(),
            predictions=predictions,
            metric_results=metric_results,
            timestamp_iso=timestamp,
            duration_s=round(total_s, 3),
        )

        self._export(run_result)
        logger.info("EvaluationRunner.run done | duration_s=%.2f", total_s)
        return run_result

    def _evaluate_example(self, ex: EvaluationExample) -> EvaluationPrediction:
        monitor = PerformanceMonitor(
            capture_resources=self._config.capture_resources,
            gpu_monitoring=self._config.gpu_monitoring,
        )
        errors: List[str] = []
        retrieved_contexts: List[RetrievedContext] = []
        generated_answer: Optional[GeneratedAnswer] = None

        monitor.begin("end_to_end")

        if self._config.mode in {"retrieval", "end_to_end"}:
            monitor.begin("retrieval")
            try:
                raw_results = self._retriever.retrieve(  # type: ignore[union-attr]
                    ex.query, k=self._config.top_k
                )
                retrieved_contexts = [
                    RetrievedContext(
                        document_id=r["document_id"],
                        chunk_id=r["chunk_id"],
                        text=r["text"],
                        score=r["score"],
                        rank=i,
                        metadata={
                            k: v for k, v in r.items()
                            if k not in {"document_id", "chunk_id", "text", "score"}
                        },
                    )
                    for i, r in enumerate(raw_results)
                ]
            except Exception as exc:
                errors.append(f"retrieval error: {exc}")
                logger.warning("Retrieval failed for query '%s': %s", ex.query, exc)
            monitor.end("retrieval")

        if self._config.mode in {"generation", "end_to_end"}:
            context_texts = [ctx.text for ctx in retrieved_contexts]
            monitor.begin("generation")
            try:
                gen_result = self._strategy.generate(  # type: ignore[union-attr]
                    ex.query, context_texts
                )
                generated_answer = GeneratedAnswer(
                    text=gen_result.answer,
                    model=gen_result.model,
                    strategy=gen_result.strategy,
                    latency_ms=gen_result.latency_ms,
                    prompt_chars=gen_result.prompt_chars,
                    context_chars=gen_result.context_chars,
                    error=gen_result.error,
                    raw_response=dict(gen_result.raw_response) if gen_result.raw_response else None,
                )
                if gen_result.error:
                    errors.append(f"generation error: {gen_result.error}")
            except Exception as exc:
                errors.append(f"generation error: {exc}")
                logger.warning("Generation failed for query '%s': %s", ex.query, exc)
            monitor.end("generation")

        monitor.end("end_to_end")

        return EvaluationPrediction(
            example=ex,
            retrieved_contexts=retrieved_contexts,
            generated_answer=generated_answer,
            timings=monitor.stage_timing(),
            resource_snapshots=monitor.snapshots(),
            errors=errors,
        )

    def _build_metrics(self, total_duration_s: float) -> List[BaseMetric]:
        if self._custom_metrics is not None:
            return self._custom_metrics + self._extra_metrics
        metrics = _build_default_metrics(self._config, total_duration_s)
        return metrics + self._extra_metrics

    def _config_dict(self) -> dict:
        return {
            "mode": self._config.mode,
            "top_k": self._config.top_k,
            "retrieval_metrics": self._config.retrieval_metrics,
            "generation_metrics": self._config.generation_metrics,
            "performance_metrics": self._config.performance_metrics,
            "capture_resources": self._config.capture_resources,
            "gpu_monitoring": self._config.gpu_monitoring,
        }

    def _export(self, run_result: EvaluationRunResult) -> None:
        """Write results to JSONL/CSV if configured."""
        if self._config.output_dir is None:
            return
        output_dir = self._config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        from rag.evaluation.storage.jsonl_store import EvaluationJSONLStore
        from rag.evaluation.storage.csv_store import EvaluationCSVStore

        if self._config.export_jsonl:
            store = EvaluationJSONLStore(output_dir / f"{run_result.run_id}.jsonl")
            store.write_run_result(run_result)

        if self._config.export_csv:
            store_csv = EvaluationCSVStore(output_dir / f"{run_result.run_id}.csv")
            store_csv.write_run_result(run_result)
