# Dense Retrieval and Embedding Models

## Overview

Dense retrieval, also referred to as dense passage retrieval (DPR), is a family of retrieval methods that represent both queries and documents as fixed-dimensional dense vectors in a continuous embedding space. Relevance is measured as a similarity score between the query vector and document vectors, typically using cosine similarity, dot product, or negative L2 distance.

Unlike sparse retrieval methods such as BM25 that rely on exact term overlap, dense retrieval generalises to synonyms, paraphrases, and semantic reformulations by learning a shared semantic representation.

## Dense Passage Retrieval (DPR)

The foundational dense retrieval model is DPR, introduced by Karpukhin et al. in 2020. DPR trains two separate BERT-based encoders:

- A **question encoder** that maps queries to 768-dimensional vectors.
- A **passage encoder** that maps document passages to 768-dimensional vectors.

Both encoders are trained end-to-end using in-batch negative contrastive learning, maximising the inner product between query vectors and their corresponding relevant passage vectors while minimising the inner product with irrelevant passages.

## Modern Embedding Models

Several embedding models are commonly used for dense retrieval in RAG systems:

### BGE-M3

BGE-M3 (BAAI General Embedding, Multi-lingual, Multi-granularity, Multi-functionality) is a state-of-the-art embedding model released by BAAI. It supports dense, sparse (lexical), and multi-vector (ColBERT-style) retrieval simultaneously from a single model. Its native dense dimension is 1024 and it supports sequences up to 8192 tokens. BGE-M3 should be loaded with use_fp16=True for GPU inference.

### EmbeddingGemma

EmbeddingGemma is a 300 million parameter embedding model based on the Gemma architecture. It has a native dimension of 768 and supports Matryoshka Representation Learning (MRL), allowing the embedding dimension to be reduced to 512, 256, or 128 without significant quality loss. EmbeddingGemma requires bfloat16 precision; do not use float16, as this causes numerical overflow with the Gemma architecture.

## Matryoshka Representation Learning (MRL)

MRL trains models to produce embeddings where the first d dimensions form a valid, self-contained representation for any d in a predefined set. This allows a single model to serve at multiple embedding dimensions, trading off index size and search latency against retrieval quality.

## Retrieval Modes

Modern embedding models support multiple retrieval modes:

- **Dense**: All retrieval is done via dense vector similarity.
- **Sparse**: Lexical weights produced by the model (not BM25) are used for retrieval.
- **Hybrid**: Both dense and sparse signals are computed and fused.

## Index Search

Approximate Nearest Neighbour (ANN) search is used to find the top-k closest vectors efficiently:

- **Brute force**: Exact search, O(n) per query. Suitable for small corpora (< 10,000 documents).
- **FAISS Flat**: Exact search via Facebook AI Similarity Search, optimised for batched GPU queries.
- **FAISS HNSW**: Approximate search with O(log n) query complexity. Suitable for millions of documents.
- **FAISS IVF**: Inverted file index with approximate search, requires training on a sample of the corpus.

## Normalisation

For cosine similarity search, embedding vectors should be L2-normalised before indexing. After normalisation, cosine similarity equals the inner product, allowing efficient inner product search to substitute for cosine search. This is a standard practice in dense retrieval systems.
