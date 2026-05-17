# RAG Evaluation Metrics

## Overview

Evaluating a RAG system requires measuring both retrieval quality (did the retriever surface the right documents?) and generation quality (did the language model produce a correct answer?). These two aspects are measured by distinct families of metrics.

Performance metrics (latency, throughput, memory usage) complement quality metrics by characterising the system's efficiency under realistic operating conditions.

## Retrieval Metrics

### Context Precision

Context Precision measures the proportion of retrieved contexts that are relevant to the query:

```
Precision@k = |relevant ∩ retrieved[:k]| / |retrieved[:k]|
```

A high precision score indicates that the retriever returns mostly relevant documents and avoids injecting noise into the generator's context. Precision is always reported alongside the top-k value, since precision@1 ≥ precision@10 by definition.

### Context Recall

Context Recall measures what fraction of the known relevant documents were retrieved:

```
Recall@k = |relevant ∩ retrieved[:k]| / |relevant|
```

High recall indicates that the retriever successfully surfaces all relevant information. Recall and precision are in tension: increasing top-k improves recall but decreases precision.

### Mean Reciprocal Rank (MRR)

MRR evaluates the rank of the first relevant result across all queries:

```
RR(q) = 1 / rank_of_first_relevant_result   (0 if none found)
MRR   = mean(RR(q)) over all queries
```

Ranks are 1-indexed. MRR = 1.0 means the relevant document is always at rank 1. MRR = 0.5 means the first relevant document appears at rank 2 on average.

MRR is sensitive to the rank of the first relevant document and ignores subsequent relevant documents. For RAG, MRR above 0.5 indicates the retriever consistently surfaces relevant content near the top.

### Normalised Discounted Cumulative Gain (nDCG)

nDCG measures the full ranking quality, with higher credit given to relevant results appearing at higher ranks:

```
DCG@k  = Σ_{i=1}^{k} rel_i / log₂(i + 1)
IDCG@k = DCG of the ideal ranking (all relevant results first)
nDCG@k = DCG@k / IDCG@k
```

With binary relevance (rel_i = 1 if relevant, 0 otherwise), nDCG ranges from 0 to 1. A score of 1.0 indicates an ideal ranking. nDCG is the most comprehensive single retrieval quality metric.

## Generation Metrics

### Exact Match (EM)

Exact Match measures the proportion of predictions that exactly match the expected answer after normalisation:

```
EM = (number of exact matches) / (number of examples)
```

Normalisation typically includes: case folding, whitespace collapse, and optionally punctuation removal.

EM is a strict metric suitable for short, unambiguous factoid answers. For instruction-tuned LLMs that produce verbose responses, EM tends to be lower than Token F1 even when the answer is semantically correct.

### Token F1

Token F1 measures word-level overlap between the generated and expected answer:

```
Precision = |gen_tokens ∩ exp_tokens| / |gen_tokens|
Recall    = |gen_tokens ∩ exp_tokens| / |exp_tokens|
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Token counts are multisets; duplicate tokens are counted separately. Tokenisation uses lowercase word-boundary regex matching.

Token F1 is more lenient than EM and correlates better with human judgements for conversational or verbose model outputs.

## Performance Metrics

### Latency

Wall-clock time per pipeline stage in milliseconds. Key stages:
- Retrieval latency: Time to query the index and return candidates.
- Generation latency: Time for LLM inference, dominated by token count.
- End-to-end latency: Total query-to-answer time.

For statistical reliability, latency should be measured over at least 20 iterations with at least 1 warmup iteration excluded from reported statistics.

### Throughput

Number of examples processed per second:

```
Throughput (examples/s) = n_examples / total_duration_s
```

### Memory Usage

Peak Resident Set Size (RSS) of the process in megabytes, measured from OS-level snapshots during the evaluation run. Requires psutil.

## Why LLM-as-a-Judge Is Excluded

This evaluation framework intentionally excludes LLM-as-a-judge metrics for three reasons:

1. **Reproducibility**: LLM judge outputs are non-deterministic even at temperature=0 and change across model versions.
2. **No external API**: The framework operates entirely locally without commercial API dependencies.
3. **Scientific interpretability**: Deterministic metrics have well-defined mathematical formulae that can be verified by inspection.
