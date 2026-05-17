# Benchmarking Guide

## Purpose

Benchmarks measure the **performance characteristics** of the RAG system under
controlled, reproducible conditions. They are distinct from evaluation:

| Concern           | Evaluation              | Benchmarking              |
|-------------------|-------------------------|---------------------------|
| Question          | Is the answer correct?  | How fast is the system?   |
| Input             | Labelled dataset        | Query pool + callable     |
| Output            | Metric scores           | Latency statistics        |
| Statistical unit  | Examples                | Iterations                |

## Core Concepts

### Warmup iterations

Local LLM systems (Ollama/Mistral) load model weights into RAM or VRAM on the
first inference call. This cold-start latency can be 10–100× the steady-state
latency and must not contaminate measured results.

Always set `warmup_iterations >= 1` before measuring generation latency.

### Measured iterations

The benchmark records `n_iterations` measured calls after warmup. Increase this
until the coefficient of variation (`std_dev / mean`) is below 10%.

Rule of thumb: at least 20 iterations for retrieval; at least 10 for generation.

### Statistics reported

| Statistic | Description                                                      |
|-----------|------------------------------------------------------------------|
| `mean`    | Average latency. Sensitive to outliers.                          |
| `median`  | Middle value. Robust to outliers; good for skewed distributions. |
| `min`     | Best-case latency.                                               |
| `max`     | Worst-case latency; often dominated by cold-start or GC.         |
| `p95`     | 95th percentile.                                                 |
| `std_dev` | Standard deviation. High value means noisy results.              |

Prefer median and p95 over mean when the distribution is right-skewed.

## Running a Retrieval Benchmark

```python
from rag.evaluation.benchmarks import RetrievalBenchmark, BenchmarkConfig
from rag.evaluation.report import BenchmarkReport

config = BenchmarkConfig(
    name="retrieval_latency_top10",
    n_iterations=50,
    warmup_iterations=5,
    top_k=10,
)

bench = RetrievalBenchmark(
    config=config,
    retriever=your_retrieval_orchestrator,
    queries=["What is RAG?", "How does BM25 work?", "What is FAISS?"],
    top_k=10,
    corpus_size=1000,
)

result = bench.run()
print(BenchmarkReport(result).summary())
```

## Running a Generation Benchmark

```python
from rag.evaluation.benchmarks import GenerationBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    name="generation_latency_mistral7b",
    n_iterations=20,
    warmup_iterations=3,
    model_name="mistral:7b-instruct-q4_K_M",
)

bench = GenerationBenchmark(
    config=config,
    strategy=your_generation_strategy,
    queries=["What is RAG?"],
    context_chunks=[["RAG stands for Retrieval-Augmented Generation."]],
)

result = bench.run()
print(BenchmarkReport(result).summary())
```

## Running an End-to-End Benchmark

```python
from rag.evaluation.benchmarks import EndToEndBenchmark, BenchmarkConfig

config = BenchmarkConfig(
    name="end_to_end_latency",
    n_iterations=20,
    warmup_iterations=2,
    top_k=10,
    model_name="mistral",
)

bench = EndToEndBenchmark(
    config=config,
    retriever=your_retrieval_orchestrator,
    strategy=your_generation_strategy,
    queries=["What is RAG?", "Explain BM25."],
    top_k=10,
)

result = bench.run()
```

## Running a Corpus-Size Scaling Benchmark

```python
from rag.evaluation.benchmarks import CorpusScalingExperiment, BenchmarkConfig, ScalingConfig

base_config = BenchmarkConfig(name="corpus_scaling", n_iterations=30, warmup_iterations=3)
scaling = ScalingConfig(corpus_sizes=[100, 500, 1000, 5000])

experiment = CorpusScalingExperiment(
    base_config=base_config,
    retriever=your_retrieval_orchestrator,
    queries=my_query_pool,
    scaling_config=scaling,
    top_k=10,
)

results = experiment.run()
for r in results:
    print(f"Corpus {r.corpus_size}: mean={r.stats.mean:.1f}ms p95={r.stats.p95:.1f}ms")
```

## Running a Concurrency Benchmark

```python
from rag.evaluation.benchmarks import ConcurrencyExperiment, BenchmarkConfig, ScalingConfig

config = BenchmarkConfig(name="concurrency", n_iterations=60, warmup_iterations=3)
scaling = ScalingConfig(concurrency_levels=[1, 2, 4, 8])

experiment = ConcurrencyExperiment(
    base_config=config,
    callable_fn=lambda: your_retrieval_orchestrator.retrieve("What is RAG?", k=10),
    scaling_config=scaling,
)

results = experiment.run()
for r in results:
    throughput = r.measured_n / r.duration_s
    print(f"Concurrency {r.concurrency}: {throughput:.1f} req/s, p95={r.stats.p95:.1f}ms")
```

## Interpreting Statistics

Use the median when latency has a long tail (typical for local LLMs). The mean
is pulled upward by outliers (GC pauses, OS scheduling, cold model starts).

`std_dev / mean > 0.5` indicates highly variable results. Common causes: shared
HPC load, insufficient warmup, OS memory pressure, garbage collection pauses.

## Avoiding Misleading Conclusions

1. Always report k, corpus size, concurrency level, and model name.
2. Keep warmup and measured runs separate.
3. Do not compare benchmarks across different hardware.
4. Do not extrapolate small-corpus trends to larger sizes.
5. Be explicit about which component is measured.
6. Record hostname, CPU count, and RAM in the benchmark result metadata.

## Saving Benchmark Results

```python
from rag.evaluation.storage.jsonl_store import EvaluationJSONLStore
from pathlib import Path

store = EvaluationJSONLStore(Path("results/benchmarks.jsonl"))
store.write_benchmark(result)
```
