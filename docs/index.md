# RAG Toolkit

A local Retrieval-Augmented Generation toolkit with clean, testable layers
for ingestion, embedding, indexing, retrieval, generation, and evaluation.

## Highlights

- **Local-first.** Runs on Ollama for generation and on sentence-transformers /
  FlagEmbedding for embeddings. No external APIs required.
- **Modular layers.** Each layer has a narrow contract; layers can be swapped,
  tested, and benchmarked independently.
- **Reproducible evaluation.** Deterministic metrics (Context Precision/Recall,
  MRR, nDCG, Exact Match, Token F1) plus a separate benchmarking layer for
  latency, throughput, and scaling experiments.
- **Hybrid retrieval.** Dense + sparse (BM25, TF-IDF, learned sparse) with
  configurable fusion (weighted sum, RRF).
- **uv-friendly.** Project metadata in `pyproject.toml`; install with `uv sync`.

## Quick example

```python
from rag.evaluation import (
    EvaluationConfig, EvaluationRunner, EvaluationReport, dataset_from_dicts,
)

config = EvaluationConfig(mode="end_to_end", top_k=5)
examples = dataset_from_dicts([
    {
        "query": "What is RAG?",
        "expected_answer": "Retrieval-Augmented Generation",
        "relevant_document_ids": ["doc_abc123"],
    }
])
runner = EvaluationRunner(config, retriever=my_retriever, strategy=my_strategy)
result = runner.run(examples)
print(EvaluationReport(result).summary())
```

## Next steps

- [Getting started](getting-started.md) — install, run, and evaluate.
- [API reference](api.md) — every public module documented.
- [Development](development.md) — building docs and running tests locally.
