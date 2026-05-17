"""End-to-end RAG pipeline latency benchmark.

Measures wall-clock time from query to generated answer (retrieval + optional
reranking + prompt construction + generation).
"""

from __future__ import annotations

import itertools
from typing import List, Optional

from rag.evaluation.benchmarks.benchmark_case import BenchmarkCase
from rag.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from rag.evaluation.benchmarks.config import BenchmarkConfig
from rag.evaluation.types import BenchmarkResult


class EndToEndBenchmark:
    """Benchmark the full RAG pipeline latency."""

    def __init__(
        self,
        config: BenchmarkConfig,
        retriever,
        strategy,
        queries: List[str],
        top_k: Optional[int] = None,
        corpus_size: Optional[int] = None,
    ) -> None:
        if not queries:
            raise ValueError("EndToEndBenchmark requires at least one query.")
        self._config = config
        self._retriever = retriever
        self._strategy = strategy
        self._queries = queries
        self._top_k = top_k or config.top_k
        self._corpus_size = corpus_size

    def run(self) -> BenchmarkResult:
        """Execute the end-to-end benchmark and return statistics."""
        query_cycle = itertools.cycle(self._queries)

        def _end_to_end() -> None:
            query = next(query_cycle)
            results = self._retriever.retrieve(query, k=self._top_k)
            context_texts = [r["text"] for r in results]
            self._strategy.generate(query, context_texts)

        case = BenchmarkCase(
            name=self._config.name,
            fn=_end_to_end,
            metadata={
                "benchmark_type": "end_to_end",
                "n_queries_in_pool": len(self._queries),
                "top_k": self._top_k,
                "model_name": self._config.model_name or "unknown",
            },
        )

        runner = BenchmarkRunner(self._config)
        return runner.run(
            case,
            corpus_size=self._corpus_size,
            top_k=self._top_k,
        )
