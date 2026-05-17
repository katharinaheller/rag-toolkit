# RAG Evaluation Layer

## Purpose

`rag.evaluation` measures the behaviour of the RAG pipeline along three axes:

1. Does retrieval surface the right documents? (retrieval quality)
2. Does generation produce correct answers? (answer quality)
3. Is the system fast enough and resource-efficient? (performance)

The evaluation layer is read-only with respect to the rest of the pipeline.

## Architectural Position

```
rag/
  ingestion/      loads, cleans, chunks documents
  embedding/      computes dense and sparse vectors
  indexing/       builds searchable indexes
  retrieval/      scores and ranks candidates
  generation/     prompts the local LLM and returns answers
  evaluation/     ← measures all of the above
    metrics/      individual quality and performance metrics
    monitors/     timing and resource snapshots
    benchmarks/   repeatable performance experiments
    storage/      persist evaluation and benchmark results
    report.py     human-readable summaries
```

## Responsibilities

The evaluation layer is responsible for:

- Accepting a labelled dataset of `EvaluationExample` objects.
- Running the configured RAG pipeline on each example.
- Measuring per-stage and end-to-end latency.
- Optionally collecting CPU and RAM snapshots.
- Computing retrieval and generation metrics.
- Persisting results to JSONL and CSV.
- Producing human-readable summaries.
- Running benchmarks and scaling experiments.

The evaluation layer is NOT responsible for:

- Building or rebuilding the index.
- Embedding documents or queries.
- Choosing retrieval strategies or prompt templates.
- Labelling or annotation.
- Calling external APIs.

## Modes

| Mode         | Runs retrieval? | Runs generation? | Typical use case                      |
|--------------|-----------------|------------------|---------------------------------------|
| `retrieval`  | Yes             | No               | Tune retrieval settings, test indexes |
| `generation` | No              | Yes              | Test prompts with pre-retrieved ctx   |
| `end_to_end` | Yes             | Yes              | Full system evaluation                |

## Running Retrieval-Only Evaluation

```python
from pathlib import Path
from rag.evaluation import EvaluationConfig, EvaluationRunner, EvaluationReport
from rag.evaluation.dataset import load_jsonl_dataset

examples = load_jsonl_dataset(Path("data/eval_dataset.jsonl"))

config = EvaluationConfig(
    mode="retrieval",
    top_k=10,
    retrieval_metrics=["context_precision", "context_recall", "mrr", "ndcg"],
    performance_metrics=["latency"],
    output_dir=Path("results/"),
)

runner = EvaluationRunner(config, retriever=your_retrieval_orchestrator)
result = runner.run(examples)

print(EvaluationReport(result).summary())
```

## Running End-to-End Evaluation

```python
from rag.evaluation import EvaluationConfig, EvaluationRunner

config = EvaluationConfig(
    mode="end_to_end",
    top_k=10,
    retrieval_metrics=["context_precision", "context_recall", "mrr", "ndcg"],
    generation_metrics=["exact_match", "token_f1"],
    performance_metrics=["latency", "throughput", "memory_usage"],
    output_dir=Path("results/"),
    capture_resources=True,
)

runner = EvaluationRunner(
    config,
    retriever=your_retrieval_orchestrator,
    strategy=your_generation_strategy,
)
result = runner.run(examples)
```

## Example Dataset (JSONL)

```json
{"query": "What is RAG?", "expected_answer": "Retrieval-Augmented Generation", "relevant_document_ids": ["doc_abc123"]}
{"query": "What does MRR measure?", "expected_answer": "Mean Reciprocal Rank", "relevant_document_ids": ["doc_def456"], "relevant_chunk_ids": ["chunk_789"]}
```

## Configuration Reference

| Field                | Type         | Default         | Description                                |
|----------------------|--------------|-----------------|--------------------------------------------|
| `mode`               | str          | `"end_to_end"`  | Evaluation scope                           |
| `top_k`              | int          | `10`            | Retrieved contexts per query               |
| `retrieval_metrics`  | List[str]    | All retrieval   | Retrieval metrics to compute               |
| `generation_metrics` | List[str]    | All generation  | Generation metrics to compute              |
| `performance_metrics`| List[str]    | All performance | Performance metrics to compute             |
| `output_dir`         | Path or None | None            | Result output directory                    |
| `export_jsonl`       | bool         | True            | Write JSONL result file                    |
| `export_csv`         | bool         | True            | Write CSV result file                      |
| `capture_resources`  | bool         | False           | Collect CPU/RAM snapshots (requires psutil)|
| `gpu_monitoring`     | bool         | False           | Attempt GPU metric collection (pynvml)     |
| `run_id`             | str or None  | Auto-generated  | Stable identifier for this run             |
