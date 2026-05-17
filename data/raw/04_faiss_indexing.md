# FAISS: Facebook AI Similarity Search

## Overview

FAISS, which stands for Facebook AI Similarity Search, is an open-source library developed by Facebook AI Research (now Meta AI Research) for efficient similarity search and clustering of dense vectors. FAISS is written in C++ with Python bindings and is one of the most widely used libraries for building large-scale vector search systems.

FAISS supports both exact and approximate nearest neighbour search, GPU acceleration, and a variety of indexing strategies suited to different corpus sizes and latency requirements.

## Installation

FAISS is available in two variants:

```
pip install faiss-cpu   # CPU-only version
pip install faiss-gpu   # GPU-accelerated version (requires CUDA)
```

## Index Types

### IndexFlatL2 and IndexFlatIP

The flat indexes perform exact brute-force search. `IndexFlatL2` computes L2 (Euclidean) distance; `IndexFlatIP` computes inner product similarity. These indexes require no training and support incremental vector addition. Their query time complexity is O(n) per query.

For cosine similarity search, vectors should be L2-normalised before insertion and query, after which `IndexFlatIP` produces exact cosine similarity scores.

### IndexHNSWFlat (HNSW)

Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest neighbour index. HNSW maintains a multi-layer graph of connections between vectors. Query time complexity is O(log n), making it highly scalable for large corpora. HNSW does not require a training phase.

Key parameters:
- **M**: Number of connections per node (default: 32). Higher values improve recall but increase memory and build time.
- **efConstruction**: Size of the dynamic candidate list during construction (default: 200). Higher values improve index quality but slow construction.
- **efSearch**: Size of the candidate list during search. Can be set at query time to trade off speed and recall.

HNSW is the recommended index for corpora with 10,000 to 100,000,000 documents when GPU is not available.

### IndexIVFFlat (IVF)

Inverted File Index clusters vectors into nlist Voronoi cells. During search, only nprobe nearest cells are examined. IVF requires a training phase on a representative sample of the corpus (at least 10 × nlist vectors are recommended).

Query time complexity is approximately O(n / nlist) per query. IVF is suitable for very large corpora where memory efficiency is critical.

## Memory Requirements

FAISS stores vectors in 32-bit float precision. Memory consumption:

```
Memory (bytes) ≈ n_vectors × dimension × 4
```

For 1 million 1024-dimensional BGE-M3 vectors:
```
1,000,000 × 1024 × 4 ≈ 4 GB
```

HNSW adds approximately `M × 8` bytes per vector for the graph structure, so a corpus of 1 million vectors with M=32 requires approximately 4.25 GB.

## GPU Acceleration

FAISS supports GPU-accelerated flat and IVF indexes. GPU search can be 10–100× faster than CPU search for large batches. GPU indexes are instantiated via `faiss.StandardGpuResources` and `faiss.index_cpu_to_gpu`.

GPU acceleration is most beneficial for large corpora and large query batches. For small corpora or single-query workloads, CPU HNSW is typically faster due to GPU initialisation overhead.

## Persistence

FAISS indexes can be serialised to disk:

```python
faiss.write_index(index, "index.faiss")
index = faiss.read_index("index.faiss")
```

The index file includes all vectors and graph structure. An additional ID mapping file is typically needed to associate FAISS integer IDs with external document identifiers.

## Choosing an Index

| Corpus Size | Recommended Index | Notes |
|-------------|------------------|-------|
| < 10,000 | BruteForce or IndexFlatIP | Exact, no training needed |
| 10,000–1,000,000 | IndexHNSWFlat | Approximate, O(log n), no training |
| > 1,000,000 | IndexIVFFlat | Training required, memory efficient |
