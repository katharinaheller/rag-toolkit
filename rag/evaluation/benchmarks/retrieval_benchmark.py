"""Retrieval latency benchmark. Queries are cycled from the provided pool."""

from __future__ import annotations

import itertools
from typing import List, Optional

from rag.evaluation.benchmarks.benchmark_case import BenchmarkCase
from rag.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from rag.evaluation.benchmarks.config import BenchmarkConfig
from rag.evaluation.types import BenchmarkResult


class RetrievalBenchmark:
    """Benchmark retrieval latency for a RetrievalOrchestrator."""

    def __init__(
        self,
        config: BenchmarkConfig,
        retriever,
        queries: List[str],
        top_k: Optional[int] = None,
        corpus_size: Optional[int] = None,
    ) -> None:
        if not queries:
            raise ValueError("RetrievalBenchmark requires at least one query.")
        self._config = config
        self._retriever = retriever
        self._queries = queries
        self._top_k = top_k or config.top_k
        self._corpus_size = corpus_size

    def run(self) -> BenchmarkResult:
        """Execute the retrieval benchmark and return statistics."""
        query_cycle = itertools.cycle(self._queries)

        def _retrieve() -> None:
            query = next(query_cycle)
            self._retriever.retrieve(query, k=self._top_k)

        case = BenchmarkCase(
            name=self._config.name,
            fn=_retrieve,
            metadata={
                "benchmark_type": "retrieval",
                "n_queries_in_pool": len(self._queries),
            },
        )

        runner = BenchmarkRunner(self._config)
        return runner.run(
            case,
            corpus_size=self._corpus_size,
            top_k=self._top_k,
        )
