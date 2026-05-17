# Embedding Models for RAG

## Overview

Embedding models are neural networks that map text sequences to fixed-dimensional continuous vectors. In the context of RAG, embedding models serve two roles:

1. **Document embedding**: Convert document chunks into vectors at index build time.
2. **Query embedding**: Convert user queries into vectors at inference time.

The quality of the embedding model is the primary determinant of retrieval quality in dense and hybrid RAG systems.

## Key Properties of Embedding Models

### Semantic Similarity

A good embedding model assigns similar vectors to semantically equivalent texts, even when they use different vocabulary. For example:
- "What is the speed of light?" and "How fast does light travel?" should produce similar embeddings.

This generalisation capability is the main advantage of dense retrieval over BM25.

### Embedding Dimension

Embedding dimension determines the expressiveness of the vector space and the storage and compute requirements of the index. Common dimensions:

- 384 dimensions: Small models (MiniLM, all-mpnet-base-v2)
- 768 dimensions: BERT-scale models, EmbeddingGemma
- 1024 dimensions: BGE-M3, Cohere Embed v3

Higher dimensions generally improve retrieval quality at the cost of increased memory and compute.

### Maximum Sequence Length

Embedding models have a maximum sequence length, beyond which text is truncated. Common values:

- 512 tokens: Most BERT-based models
- 4096 tokens: Modern long-context models
- 8192 tokens: BGE-M3

For RAG, chunks should be sized so that their token length is below the model's maximum sequence length. Using 75% of the maximum is a safe heuristic.

### Query and Document Asymmetry

Many modern embedding models use different encodings for queries and documents. For BGE models, queries are typically prefixed with a task-specific instruction. EmbeddingGemma uses distinct template strings for queries ("task: search result | query: ...") and documents ("title: | text: ...").

Failing to apply the correct prefix to queries or documents will substantially degrade retrieval quality.

## Model Selection Guide

### BGE-M3 (BAAI/bge-m3)

**Type**: Multi-function (dense + sparse + multi-vector)
**Dimension**: 1024
**Max sequence length**: 8192 tokens
**Precision**: float16 (use_fp16=True) on GPU; float32 on CPU
**Languages**: 100+ languages
**Strengths**: Hybrid retrieval support; long context; multilingual; strong BEIR performance
**Weaknesses**: Requires FlagEmbedding library; ~4 GB RAM in float16

BGE-M3 is the recommended model for production RAG systems where hybrid retrieval is desired.

### EmbeddingGemma (google/embedding-gemma-300m)

**Type**: Dense only
**Dimension**: 768 (with MRL support at 512, 256, 128)
**Max sequence length**: 2048 tokens
**Precision**: bfloat16 (use_bfloat16=True); do NOT use float16
**Languages**: Primarily English
**Strengths**: MRL support; high quality for English; ~1.2 GB RAM in bfloat16
**Weaknesses**: Dense only; bfloat16 precision requirement

### Sentence-BERT Models (sentence-transformers)

A large family of models available via the sentence-transformers library. Common choices:
- `all-MiniLM-L6-v2`: 384 dimensions, fast, good quality
- `all-mpnet-base-v2`: 768 dimensions, high quality, slower
- `multi-qa-mpnet-base-dot-v1`: Optimised for question-answer retrieval

## Normalisation

For cosine similarity retrieval, embedding vectors should be L2-normalised. Most modern embedding models (BGE-M3, sentence-transformers) produce normalised vectors by default when normalise=True. Verify that normalisation is applied consistently at both index build time and query time.

## Model Caching

Loading embedding models is expensive (seconds to minutes depending on hardware). Production RAG systems should cache loaded models in memory and reuse them across requests. A thread-safe LRU cache with a configurable maximum number of cached models prevents redundant loading while bounding memory usage.
