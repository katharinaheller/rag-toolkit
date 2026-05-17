# API Reference

Documentation is generated directly from the source code via
[mkdocstrings](https://mkdocstrings.github.io/) using the Python handler.

The reference is split by layer; each page mirrors a top-level package:

- [Ingestion](api/ingestion.md) — loading, cleaning, chunking, persistence.
- [Embedding](api/embedding.md) — dense and sparse embedding orchestration.
- [Indexing](api/indexing.md) — dense (BruteForce, FAISS) and sparse indexes.
- [Retrieval](api/retrieval.md) — dense, BM25, TF-IDF, learned sparse, hybrid.
- [Generation](api/generation.md) — Ollama client, prompt builder, strategies.
- [Evaluation](api/evaluation.md) — metrics, monitors, benchmarks, reports.

## Conventions

- Configs are immutable frozen dataclasses.
- Streaming pipelines use generators wherever possible.
- IDs are deterministic (SHA-256 over natural keys or content).
- Errors raised in inner loops are captured into `EvaluationPrediction.errors`
  rather than aborting the whole run.
