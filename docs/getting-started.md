# Getting Started

This page gets you from a fresh clone to a working install, the documentation
site, and a first experiment run. It assumes no prior knowledge of the internal
implementation.

## Run all commands from the repository root

Every command below is executed from the repository root — the folder that
contains `pyproject.toml`:

```bash
cd rag-toolkit
```

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for environment and dependency management
- A running Ollama instance for generation (optional — retrieval-only workflows
  do not need it)

## Install

```bash
uv sync
```

`uv sync` installs the runtime dependencies plus the default `dev` group. Two
further dependency groups are installed explicitly when you need them:

```bash
uv sync --group dev      # test/lint tooling (pytest, pylint, hypothesis, …)
uv sync --group docs     # mkdocstrings — required to build the documentation
```

There are no `--extra` optional-dependency sets; the project uses dependency
groups (`dev`, `docs`, and an opt-in `gpu` group) exclusively.

## Run the documentation locally

The documentation site uses `mkdocstrings`, which lives in the `docs` group.
Install that group, then serve or build with `--group docs` so the plugin is
available:

```bash
uv sync --group docs
uv run --group docs mkdocs serve     # http://127.0.0.1:8000/
```

To build the static site into `site/`:

```bash
uv run --group docs mkdocs build
```

Plain `uv run mkdocs serve` (without `--group docs`) fails with a
plugin-not-found error, because `mkdocstrings` is not part of the default sync.

## Minimal end-to-end run (library)

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

## Your first experiment run

The `experiments/` framework runs a whole matrix of suites and writes a report.
A retrieval-only run needs no Ollama:

```bash
uv run experiments/run_all_experiments.py
```

The full workflow — the two operating modes, every suite, all options, output
locations, and runtime guidance — is documented in
[Running Experiments](experiments.md).

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

## Next steps

- [Running Experiments](experiments.md) — reproduce the full matrix and reports.
- [Command Line Interface](cli.md) — every CLI option in one place.
- [API reference](api.md) — the public surface of every layer.
- [Development](development.md) — building docs and running tests locally.
