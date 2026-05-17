"""Human-readable and machine-readable reports for evaluation and benchmark results."""

from __future__ import annotations

from typing import Any, Dict, List

from rag.evaluation.types import BenchmarkResult, EvaluationRunResult


_MIN_EXAMPLES_FOR_RELIABLE_METRICS = 10
_MIN_ITERATIONS_FOR_RELIABLE_BENCHMARK = 20


class EvaluationReport:
    """Summarises a completed EvaluationRunResult."""

    def __init__(self, run_result: EvaluationRunResult) -> None:
        self._r = run_result

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        r = self._r
        lines = [
            "=" * 60,
            "RAG Evaluation Run Summary",
            "=" * 60,
            f"Run ID     : {r.run_id}",
            f"Timestamp  : {r.timestamp_iso}",
            f"Mode       : {r.config_dict.get('mode', 'unknown')}",
            f"Examples   : {r.n_examples}",
            f"Duration   : {r.duration_s:.2f} s",
            "",
            "── Metrics ─────────────────────────────────────────────",
        ]

        for name, result in sorted(r.metric_results.items()):
            lines.append(f"  {name:<30} {result.value:.4f}")

        warnings = self._collect_warnings()
        if warnings:
            lines.append("")
            lines.append("── Warnings ─────────────────────────────────────────────")
            for w in warnings:
                lines.append(f"  ⚠  {w}")

        if r.errors:
            lines.append("")
            lines.append("── Run-level Errors ─────────────────────────────────────")
            for e in r.errors:
                lines.append(f"  ✗  {e}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a machine-readable summary dict."""
        return {
            **self._r.to_dict(),
            "warnings": self._collect_warnings(),
        }

    def _collect_warnings(self) -> List[str]:
        warnings: List[str] = []
        if self._r.n_examples < _MIN_EXAMPLES_FOR_RELIABLE_METRICS:
            warnings.append(
                f"Only {self._r.n_examples} examples evaluated. "
                f"Metrics are statistically unreliable below {_MIN_EXAMPLES_FOR_RELIABLE_METRICS}. "
                "Do not draw firm conclusions from this run."
            )
        n_errors = sum(1 for p in self._r.predictions if p.errors)
        if n_errors > 0:
            warnings.append(
                f"{n_errors}/{self._r.n_examples} predictions had captured errors. "
                "Inspect EvaluationPrediction.errors for details."
            )
        if self._r.config_dict.get("mode") == "end_to_end":
            n_no_gen = sum(1 for p in self._r.predictions if p.generated_answer is None)
            if n_no_gen > 0:
                warnings.append(
                    f"{n_no_gen} predictions have no generated answer. "
                    "Generation metrics for these examples will be 0.0."
                )
        return warnings


class BenchmarkReport:
    """Summarises a completed BenchmarkResult."""

    def __init__(self, result: BenchmarkResult) -> None:
        self._r = result

    def summary(self) -> str:
        """Return a human-readable multi-line summary string."""
        r = self._r
        s = r.stats
        lines = [
            "=" * 60,
            f"Benchmark: {r.benchmark_name}",
            "=" * 60,
            f"Timestamp  : {r.timestamp_iso}",
            f"Duration   : {r.duration_s:.2f} s",
            f"Warmup     : {r.warmup_n} iterations (excluded)",
            f"Measured   : {r.measured_n} iterations",
        ]
        if r.corpus_size is not None:
            lines.append(f"Corpus     : {r.corpus_size} documents")
        if r.top_k is not None:
            lines.append(f"Top-k      : {r.top_k}")
        if r.concurrency is not None:
            lines.append(f"Concurrency: {r.concurrency}")
        if r.model_name:
            lines.append(f"Model      : {r.model_name}")

        lines += [
            "",
            "── Latency Statistics (ms) ──────────────────────────────",
            f"  Mean   : {s.mean:.2f} ms",
            f"  Median : {s.median:.2f} ms",
            f"  Min    : {s.min:.2f} ms",
            f"  Max    : {s.max:.2f} ms",
            f"  p95    : {s.p95:.2f} ms",
            f"  StdDev : {s.std_dev:.2f} ms",
        ]

        if r.hardware_metadata:
            lines.append("")
            lines.append("── Hardware ─────────────────────────────────────────────")
            for k, v in r.hardware_metadata.items():
                lines.append(f"  {k}: {v}")

        warnings = self._collect_warnings()
        if warnings:
            lines.append("")
            lines.append("── Warnings ─────────────────────────────────────────────")
            for w in warnings:
                lines.append(f"  ⚠  {w}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a machine-readable summary dict."""
        return {
            **self._r.to_dict(),
            "warnings": self._collect_warnings(),
        }

    def _collect_warnings(self) -> List[str]:
        warnings: List[str] = list(self._r.warnings)
        if self._r.measured_n < _MIN_ITERATIONS_FOR_RELIABLE_BENCHMARK:
            warnings.append(
                f"Only {self._r.measured_n} measured iterations. "
                f"Statistics are unreliable below {_MIN_ITERATIONS_FOR_RELIABLE_BENCHMARK}. "
                "Increase benchmark_config.n_iterations for trustworthy results."
            )
        if self._r.warmup_n == 0:
            warnings.append(
                "No warmup iterations were performed. "
                "Cold-start effects may inflate mean latency. "
                "Use benchmark_config.warmup_iterations >= 1."
            )
        s = self._r.stats
        if s.n > 0 and s.std_dev > s.mean * 0.5:
            warnings.append(
                f"High variance detected (std_dev={s.std_dev:.2f} ms, mean={s.mean:.2f} ms). "
                "Results may be influenced by external system load. "
                "Consider running on a quieter system or increasing iterations."
            )
        return warnings
