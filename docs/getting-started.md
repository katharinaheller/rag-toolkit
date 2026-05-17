# Getting Started

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- A running Ollama instance for generation (optional for retrieval-only workflows)

## Install

```bash
uv sync
```

Optional dependency groups:

```bash
uv sync --extra embedding         # sentence-transformers, FlagEmbedding, torch
uv sync --extra indexing-faiss    # faiss-cpu, numpy
uv sync --extra monitoring        # psutil, pynvml
```

The documentation toolchain lives in the `docs` dependency group and is
already pulled in by `uv sync` when needed by `uv run mkdocs ...`.

## Run the documentation locally

```bash
uv run mkdocs serve
```

Then open <http://127.0.0.1:8000/>.

To build the static site into `site/`:

```bash
uv run mkdocs build
```

## Minimal end-to-end run

```python
from pathlib import Path
from rag.evaluation import (
    EvaluationConfig, EvaluationRunner, EvaluationReport, dataset_from_dicts,
)

examples = dataset_from_dicts([
    {
        "query": "What is RAG?",
        "expected_answer": "Retrieval-Augmented Generation",
        "relevant_document_ids": ["doc_abc123"],
    },
])

config = EvaluationConfig(
    mode="end_to_end",
    top_k=5,
    output_dir=Path("results"),
)

runner = EvaluationRunner(config, retriever=my_retriever, strategy=my_strategy)
result = runner.run(examples)
print(EvaluationReport(result).summary())
```

## Modes at a glance

| Mode         | Runs retrieval? | Runs generation? | Use case                         |
|--------------|-----------------|------------------|----------------------------------|
| `retrieval`  | yes             | no               | Tune indexes and retrievers      |
| `generation` | no              | yes              | Tune prompts on fixed contexts   |
| `end_to_end` | yes             | yes              | Full pipeline measurement        |

## Folder layout

```
rag/
  ingestion/      load, clean, chunk documents
  embedding/      compute dense and sparse vectors
  indexing/       build dense and sparse indexes
  retrieval/      score and rank candidates
  generation/     prompt the local LLM via Ollama
  evaluation/     metrics, monitors, benchmarks, reports
  logging/        structured logging helpers
```
