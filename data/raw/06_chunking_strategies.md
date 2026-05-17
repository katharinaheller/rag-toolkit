# Document Chunking Strategies

## Overview

Chunking is the process of splitting source documents into smaller, semantically coherent fragments called chunks before embedding and indexing. Chunking is a critical design decision in RAG pipelines: chunk size and overlap directly affect retrieval precision, recall, context quality, and index size.

All chunks produced by a RAG pipeline are assigned deterministic, collision-resistant identifiers derived from a hash of their content and position. This ensures that chunk IDs are stable across pipeline runs with the same input and strategy.

## Why Chunking Matters

Large language models have limited context windows. Even if a model supports 32,000 tokens, injecting entire documents wastes the context budget and may cause the model to lose focus on relevant content. Chunking ensures that:

1. Each retrieved passage is focused on a specific sub-topic.
2. The generator receives dense, relevant context rather than diluted whole documents.
3. The embedding model can represent the passage with high fidelity within its maximum sequence length.

## Fixed-Size Overlapping Windows

The simplest chunking strategy divides document text into fixed-character windows with a configurable overlap:

- **Chunk size**: Maximum character length of each chunk.
- **Overlap**: Number of characters shared between adjacent chunks.
- **Step**: chunk_size − overlap.

A chunk_size of 256–512 characters with 10–20% overlap is a common starting point.

**Advantages**: Simple, predictable, consistent chunk sizes.
**Disadvantages**: May split sentences or paragraphs mid-way, producing incoherent chunks.

## Structure-Aware Chunking

Structure-aware chunking respects the document's natural structure (headings, paragraphs, lists, code blocks) by parsing the Markdown AST and grouping related blocks into chunks of bounded size.

Algorithm:
1. Parse the document into a sequence of top-level blocks (headings, paragraphs, etc.).
2. Accumulate blocks into the current chunk until adding the next block would exceed max_chunk_size.
3. Emit the current chunk and start a new one.

**Advantages**: Chunks align with semantic boundaries; embeddings are more coherent.
**Disadvantages**: Chunk sizes vary, making index memory consumption less predictable.

## Chunk Size vs Retrieval Quality

The relationship between chunk size and retrieval quality is non-trivial:

| Chunk Size | Precision | Recall | Index Size |
|------------|-----------|--------|------------|
| Small (128–256 chars) | High | Low per chunk | Large |
| Medium (512 chars) | Balanced | Balanced | Moderate |
| Large (1024+ chars) | Low | High per chunk | Small |

Small chunks are more precise but require a higher top-k to cover multi-sentence answers. Large chunks cover more ground but introduce irrelevant content into the prompt.

## Chunk Identifier Strategies

Two main strategies for assigning chunk identifiers:

- **Positional chunk IDs**: Derived from (document_id, strategy, index). Stable across content changes; suitable for production pipelines where incremental updates are needed.
- **Content chunk IDs**: Include the chunk text hash. Change when content changes; useful for drift detection across pipeline versions.

## Overlap and Boundary Effects

Overlapping chunks ensure that relevant content near a chunk boundary is not lost. A passage that straddles a boundary without overlap would be retrieved with half its context missing. Overlap of 10–20% of chunk_size is standard.

Excessive overlap wastes storage and embedding compute. Overlaps above 50% are rarely beneficial.

## Practical Recommendations

For a domain-specific RAG system over markdown documentation:
- Use structure-aware chunking as the primary strategy.
- Set max_chunk_size to 400–600 characters.
- Use a fixed-overlap fallback of chunk_size=512, overlap=64 for documents that fail structure parsing.
- Evaluate with Context Precision and Recall at top-k=10.
