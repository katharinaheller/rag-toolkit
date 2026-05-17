# rag-toolkit

A local Retrieval-Augmented Generation toolkit with clean, testable layers
for ingestion, embedding, indexing, retrieval, generation, and evaluation.

## Install

```bash
uv sync
```

## Documentation

```bash
uv run mkdocs serve     # live preview on http://127.0.0.1:8000/
uv run mkdocs build     # static site under site/
```

See [`docs/getting-started.md`](docs/getting-started.md) for a minimal
end-to-end example and [`docs/api.md`](docs/api.md) for the API reference.

## Project layout

```
rag/
  ingestion/      load, clean, chunk documents
  embedding/      dense and sparse embeddings
  indexing/       dense (BruteForce, FAISS) and sparse indexes
  retrieval/      dense, BM25, TF-IDF, learned sparse, hybrid
  generation/     Ollama client, prompt templates, generation strategies
  evaluation/     metrics, monitors, benchmarks, reports
  logging/        structured logging
docs/             mkdocs source
mkdocs.yml        documentation config
pyproject.toml    project metadata and dependencies
```

## License

MIT.
