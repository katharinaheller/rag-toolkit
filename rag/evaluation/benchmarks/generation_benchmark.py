"""Generation latency benchmark.

Measures the LLM inference stage only. Context chunks are pre-prepared and
reused. Always set warmup_iterations >= 1 — local Ollama inference has a
significant cold-start cost.
"""

from __future__ import annotations

import itertools
from typing import List, Optional

from rag.evaluation.benchmarks.benchmark_case import BenchmarkCase
from rag.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from rag.evaluation.benchmarks.config import BenchmarkConfig
from rag.evaluation.types import BenchmarkResult


class GenerationBenchmark:
    """Benchmark generation latency for a BaseGenerationStrategy."""

    def __init__(
        self,
        config: BenchmarkConfig,
        strategy,
        queries: List[str],
        context_chunks: List[List[str]],
        corpus_size: Optional[int] = None,
    ) -> None:
        if not queries:
            raise ValueError("GenerationBenchmark requires at least one query.")
        if len(queries) != len(context_chunks):
            raise ValueError(
                "queries and context_chunks must have the same length."
            )
        self._config = config
        self._strategy = strategy
        self._queries = queries
        self._context_chunks = context_chunks
        self._corpus_size = corpus_size

    def run(self) -> BenchmarkResult:
        """Execute the generation benchmark and return statistics."""
        pairs = list(zip(self._queries, self._context_chunks))
        pair_cycle = itertools.cycle(pairs)

        def _generate() -> None:
            query, chunks = next(pair_cycle)
            self._strategy.generate(query, chunks)

        case = BenchmarkCase(
            name=self._config.name,
            fn=_generate,
            metadata={
                "benchmark_type": "generation",
                "n_queries_in_pool": len(self._queries),
                "model_name": self._config.model_name or "unknown",
            },
        )

        runner = BenchmarkRunner(self._config)
        return runner.run(
            case,
            corpus_size=self._corpus_size,
        )
