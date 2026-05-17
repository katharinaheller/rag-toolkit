"""Corpus scaling experiment for the RAG benchmarking layer.

Runs retrieval benchmarks at each corpus_size in ScalingConfig and records
the latency. Does not rebuild the index per size — measure that separately.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from rag.evaluation.benchmarks.benchmark_case import BenchmarkCase
from rag.evaluation.benchmarks.benchmark_runner import BenchmarkRunner
from rag.evaluation.benchmarks.config import BenchmarkConfig, ScalingConfig
from rag.evaluation.types import BenchmarkResult

logger = logging.getLogger(__name__)


class CorpusScalingExperiment:
    """Run retrieval benchmarks at increasing corpus sizes."""

    def __init__(
        self,
        base_config: BenchmarkConfig,
        retriever,
        queries: List[str],
        scaling_config: Optional[ScalingConfig] = None,
        top_k: Optional[int] = None,
    ) -> None:
        if not queries:
            raise ValueError("CorpusScalingExperiment requires at least one query.")
        self._base_config = base_config
        self._retriever = retriever
        self._queries = queries
        self._scaling_config = scaling_config or ScalingConfig()
        self._top_k = top_k or base_config.top_k

    def run(self) -> List[BenchmarkResult]:
        """Return one BenchmarkResult per corpus size."""
        results: List[BenchmarkResult] = []

        for corpus_size in self._scaling_config.corpus_sizes:
            logger.info(
                "CorpusScalingExperiment | corpus_size=%d | top_k=%d",
                corpus_size, self._top_k,
            )
            import itertools
            query_cycle = itertools.cycle(self._queries)

            def _retrieve(cycle=query_cycle) -> None:
                query = next(cycle)
                self._retriever.retrieve(query, k=self._top_k)

            import dataclasses
            config = dataclasses.replace(
                self._base_config,
                name=f"{self._base_config.name}_n{corpus_size}",
                metadata={**self._base_config.metadata, "corpus_size": corpus_size},
            )
            case = BenchmarkCase(
                name=config.name,
                fn=_retrieve,
                metadata={"corpus_size_label": corpus_size, "top_k": self._top_k},
            )
            runner = BenchmarkRunner(config)
            result = runner.run(case, corpus_size=corpus_size, top_k=self._top_k)
            results.append(result)

        return results
