# API Reference

The reference is generated directly from the source code via
[mkdocstrings](https://mkdocstrings.github.io/) using the Python handler. Only
public names are shown — anything prefixed with an underscore is treated as
internal and filtered out.

## How to read this reference

The library is split into six layers that mirror the RAG pipeline, plus a small
logging helper. Each layer is a top-level package under `rag/` and has its own
page:

| Layer                          | Package           | Responsibility                                            |
|--------------------------------|-------------------|-----------------------------------------------------------|
| [Ingestion](api/ingestion.md)  | `rag.ingestion`   | Load, clean, chunk, and persist documents                 |
| [Embedding](api/embedding.md)  | `rag.embedding`   | Dense and sparse embedding orchestration                  |
| [Indexing](api/indexing.md)    | `rag.indexing`    | Dense (BruteForce, FAISS) and sparse indexes              |
| [Retrieval](api/retrieval.md)  | `rag.retrieval`   | Dense, BM25, TF-IDF, learned sparse, and hybrid retrieval |
| [Generation](api/generation.md)| `rag.generation`  | Ollama client, prompt builder, and strategies             |
| [Evaluation](api/evaluation.md)| `rag.evaluation`  | Metrics, monitors, benchmarks, and reports                |

The layers compose left to right: ingestion feeds embedding, which feeds
indexing, which feeds retrieval, which feeds generation; evaluation measures any
stage.

## Public surface vs. internal modules

Within each layer the modules play consistent roles, so you can tell what is a
public entry point and what is an implementation detail from the name alone:

- **Entry points** — `factory`, `orchestrator`, `service`, and `*_api` modules
  are the high-level objects you call. Start here.
- **Contracts** — `base` (abstract classes / protocols), `config` (immutable
  frozen dataclasses), and `types` (data structures) define each layer's
  interface.
- **Grouped implementations** — subpackages collect concrete, swappable
  implementations behind those contracts: `indexing.backends` and
  `indexing.sparse`, `embedding.models`, `ingestion.loaders`,
  `ingestion.chunking`, `*/storage`, and `evaluation.metrics`,
  `evaluation.monitors`, `evaluation.benchmarks`.
- **Helpers** — `utils`, `normalization`, `projection`, and similar modules are
  supporting code; you rarely import them directly.

Two layers additionally curate an explicit public surface in their package
`__init__`, so you can import the common objects straight from the package:

```python
from rag.generation import (
    GenerationConfig, OllamaClient, SimpleRAGStrategy, RefineRAGStrategy,
)
from rag.evaluation import (
    EvaluationConfig, EvaluationRunner, EvaluationReport, dataset_from_dicts,
)
```

For the other layers, import from the specific module documented on its page.

## Conventions

- Configs are immutable frozen dataclasses.
- Streaming pipelines use generators wherever possible.
- IDs are deterministic (SHA-256 over natural keys or content).
- Errors raised in inner loops are captured into `EvaluationPrediction.errors`
  rather than aborting the whole run.
