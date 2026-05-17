# Hybrid Retrieval and Score Fusion

## Overview

Hybrid retrieval combines dense (semantic) retrieval with sparse (lexical) retrieval to exploit the complementary strengths of both modalities. Dense retrieval excels at capturing semantic similarity and generalising across paraphrase; sparse retrieval excels at exact term matching and handling rare entities, proper nouns, and domain-specific terminology.

Empirically, hybrid retrieval consistently outperforms either modality alone on standard benchmarks such as BEIR and MTEB, often by 3–8 percentage points in NDCG@10.

## Fusion Strategies

After obtaining ranked lists from dense and sparse retrievers, the results must be combined into a single ranking. Two main fusion strategies are in widespread use:

### Weighted Sum Fusion

The combined score for a document is a linear combination of its normalised dense and sparse scores:

```
combined_score(d) = w_dense × norm(dense_score(d)) + w_sparse × norm(sparse_score(d))
```

Min-max normalisation is applied independently to each score list before fusion:

```
norm(score) = (score - min_score) / (max_score - min_score)
```

When max_score equals min_score (all scores identical), normalised scores are set to 1.0 to preserve the retrieval leg's contribution.

The weights w_dense and w_sparse are hyperparameters that control the relative contribution of each retrieval modality. Common choices: w_dense=0.5, w_sparse=0.5 (equal weighting).

### Reciprocal Rank Fusion (RRF)

RRF is a rank-based fusion method that does not require score normalisation:

```
rrf_score(d) = Σ_r 1 / (k + rank_r(d))
```

where the sum is over all rank lists r (dense and sparse), rank_r(d) is the 1-indexed rank of document d in list r (or len(list) + 1 if absent), and k is a smoothing constant (typically k=60).

RRF has several desirable properties:
- It is insensitive to the absolute scale of relevance scores.
- It is robust to individual list quality: a low-quality list that accidentally ranks a relevant document at position 1 does not dominate the combined score.
- It consistently performs well across domains without per-domain weight tuning.

### Combination Modes

Fusion can be applied to the union or intersection of the two candidate sets:

- **Union**: All documents appearing in either list are scored. Documents absent from one list receive a rank of len(list) + 1 (RRF) or a normalised score of 0.0 (weighted sum).
- **Intersection**: Only documents appearing in both lists are scored. Union mode is more common in practice.

## Learned Sparse Retrieval

Learned sparse retrieval is distinct from lexical sparse retrieval (BM25). In learned sparse retrieval, a neural model (such as SPLADE or BGE-M3 in lexical weight mode) assigns continuous importance weights to tokens in a document or query. The retrieval score is the dot product of the query and document sparse weight vectors.

BGE-M3 produces both dense embeddings and sparse lexical weight vectors from a single forward pass. Its sparse vector contains token IDs as keys and learned weights as values. These are stored as `sparse_vector` in the document store and used directly by the learned sparse retriever.

## Index Structures for Hybrid Retrieval

A hybrid retrieval system requires:

1. **Dense index**: Stores dense vectors for approximate nearest neighbour search.
2. **Sparse index** (optional, for BM25/TF-IDF): An inverted index storing tokenised documents.
3. **Document store**: Stores the full text, metadata, dense vectors, and learned sparse vectors for each chunk. The document store is the single source of truth.

Learned sparse retrieval does not require a separate sparse index — it reads directly from the document store using an in-memory inverted index built at retrieval orchestrator initialisation time.

## Top-k Trade-offs

Increasing the retrieval top-k improves recall but decreases precision and increases context length for the generator. A typical analysis:

- Recall improves rapidly from k=1 to k=10 and then levels off.
- Precision typically decreases monotonically as k increases.
- nDCG is a balanced measure that accounts for both precision and rank quality.

The optimal k depends on the domain, corpus size, and generator context window. A top-k ablation study (evaluating at k ∈ {1, 3, 5, 10, 20}) is standard practice for RAG system characterisation.
