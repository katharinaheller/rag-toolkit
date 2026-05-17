# Performance Tuning Guide

This guide covers the RAG toolkit running in resource-constrained environments:
JupyterHub, HPC clusters, and Docker containers with limited CPU, RAM, and
optionally a shared GPU.

## CPU

Python's GIL limits true CPU parallelism for native Python code:

- Retrieval (brute-force cosine) is CPU-bound and does not benefit from
  threads. The FAISS backend releases the GIL during search.
- Generation via Ollama releases the GIL during I/O, so threads do help.

For large embedding jobs, prefer process-level parallelism. Each process loads
a full model copy, which multiplies RAM.

Larger embedding batch sizes reduce loop overhead at the cost of peak RAM.
Start at `batch_size=32`, increase until RSS reaches ~80% of available RAM,
back off on `MemoryError` or swap activity.

## RAM

| Model              | FP32 RAM | FP16 RAM  | BFloat16 RAM |
|--------------------|----------|-----------|--------------|
| BGE-M3             | ~4 GB    | ~2 GB     | N/A          |
| EmbeddingGemma 300m| ~1.2 GB  | N/A       | ~0.6 GB      |
| Mistral 7B (Q4_K_M)| ~4 GB (Ollama VRAM or RAM) | — | — |

To track RSS during evaluation:

```python
config = EvaluationConfig(
    capture_resources=True,
    performance_metrics=["memory_usage"],
)
```

Reduction strategies: `use_fp16=True` for BGE-M3; `use_bfloat16=True` for
EmbeddingGemma (do NOT use fp16 for Gemma); reduce `max_seq_length`; use
sparse-only retrieval to avoid loading a dense model.

## GPU

GPU helps for embedding large corpora, batch embedding with large batch sizes,
and long-context LLM inference. It does not help for small corpora,
single-query interactive use, or systems without a dedicated VRAM allocation.

```python
config = EmbeddingConfig(
    provider="bge-m3",
    model_name="BAAI/bge-m3",
    device="cuda",
    use_fp16=True,
    batch_size=64,
)
```

## VRAM

VRAM is not virtualised on shared GPU nodes. To check:

```python
from rag.evaluation.monitors.resource_snapshot import ResourceSnapshotCollector

collector = ResourceSnapshotCollector(gpu_monitoring=True)
snap = collector.collect()
print(snap.gpu_memory_used_mb, snap.gpu_memory_total_mb)
```

Reduction: use fp16/bfloat16, reduce batch_size, unload models via
`model_cache.evict(key)` when no longer needed, or fall back to CPU.

## Batch Size Trade-offs

| Batch size | Throughput | Peak RAM | First-batch latency |
|------------|-----------|----------|---------------------|
| 1          | Low        | Low      | Low                 |
| 32         | Good       | Medium   | Low                 |
| 128        | High       | High     | Low                 |
| 512+       | May plateau| Very high| Low                 |

On CPU, batch sizes above 64 rarely improve throughput. On GPU, 128–256 often
gives the best throughput/VRAM trade-off. Always measure for your environment.

## Corpus Size Trade-offs

| Backend           | Indexing cost | Query cost       | RAM cost       |
|-------------------|---------------|------------------|----------------|
| BruteForce        | O(1) per add  | O(n) per query   | O(n × dim)     |
| FAISS Flat        | O(1) per add  | O(n) per query   | O(n × dim)     |
| FAISS HNSW        | O(log n)      | O(log n)         | O(n × dim × m) |
| FAISS IVF         | Needs training| ~O(n/nlist)      | O(n × dim)     |

Small (< 10k): BruteForce or FAISS Flat. Medium (10k–1M): FAISS HNSW.
Large (> 1M): FAISS IVF or a dedicated vector database.

Use `CorpusScalingExperiment` to measure actual query latency before committing
to a backend.

## Retrieval Top-k

| top_k | Retrieval latency | Context chars  | Generation latency |
|-------|------------------|----------------|--------------------|
| 3     | Lowest           | Low            | Lowest             |
| 10    | Low              | Medium         | Low                |
| 50    | Higher           | High           | Higher             |

Answer quality rarely improves proportionally beyond k=10–20, but generation
cost always grows with the context.

## Chunk Size

Smaller chunks → higher precision, lower per-chunk recall, larger index,
typically needs higher top_k.

Larger chunks → higher per-chunk recall, lower precision, smaller index,
may exceed `max_context_chars`.

Typical starting point: 256–512 characters with 10–20% overlap. Evaluate at
your target top_k.

## Local LLM Latency

Ollama Mistral 7B Q4_K_M on CPU: 5–15 s cold start, ~10–30 ms per token,
512-token output ≈ 5–15 s. On consumer GPU: 1–3 s cold start, ~1–5 ms per
token, 512-token output ≈ 0.5–2 s.

Generation typically dominates total latency; retrieval is rarely the
bottleneck unless the corpus is very large.

## Ollama Bottlenecks

1. Context window: inference cost grows super-linearly with context length.
2. Concurrent requests: Ollama queues; true parallelism is GPU/CPU-limited.
3. Retry overhead: `OllamaClient` retries with fixed delay.
4. Use `temperature=0.0` with a fixed seed for reproducible evaluation.

## JupyterHub / HPC

Shared cores mean variable benchmark results. Docker memory limits can cause
OOM kills. Many JupyterHub deployments lack GPU access — check
`torch.cuda.is_available()` before setting `device="cuda"`. Add network
latency to all generation benchmarks when Ollama runs in a separate container.

## Reproducibility Limits

Even with fixed seed and temperature=0, perfect reproducibility is not
guaranteed for local LLMs due to quantisation rounding, BLAS version
differences, batch-size-dependent FP accumulation order, and thread count.

Treat results as reproducible to within a few percent on the same hardware.
Do not claim reproducibility across different hardware or Ollama versions.
