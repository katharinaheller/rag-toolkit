# Metrics Reference

## Why LLM-as-a-Judge is Excluded

LLM-as-a-judge is intentionally omitted from this toolkit:

1. No external API dependency — the toolkit runs entirely locally.
2. Reproducibility — LLM judges are non-deterministic and change with model updates.
3. Interpretability — deterministic metrics have inspectable formulae.
4. Scientific validity — a soft signal from another model has no mathematical definition.

For semantic similarity scoring, BERTScore or sentence-transformer cosine
similarity work as purely local, reproducible alternatives, and can be added
as custom `BaseMetric` implementations.

## Retrieval Metrics

### Context Precision

`rag/evaluation/metrics/context_precision.py`

```
Precision = |relevant ∩ retrieved| / |retrieved|
```

Duplicates are deduplicated before scoring. Chunk-level matching is used when
`relevant_chunk_ids` is provided; otherwise document-level matching applies.

Does not account for rank — use MRR or nDCG if rank quality matters.

### Context Recall

`rag/evaluation/metrics/context_recall.py`

```
Recall = |relevant ∩ retrieved| / |relevant|
```

Recall requires complete relevance labels. Partial annotations understate it.
Returns 0.0 with a warning when no labels are present.

### MRR — Mean Reciprocal Rank

`rag/evaluation/metrics/mrr.py`

```
RR(q) = 1 / rank_of_first_relevant_result   (0.0 if none found)
MRR   = mean(RR(q)) over all queries
```

Ranks are 1-indexed. MRR rewards systems that surface at least one correct
result quickly. It does not reward additional relevant results at lower ranks.

### nDCG — Normalised Discounted Cumulative Gain

`rag/evaluation/metrics/ndcg.py`

```
DCG@k  = Σ_{i=1}^{k}  rel_i / log₂(i + 1)
IDCG@k = DCG of the ideal ranking
nDCG@k = DCG@k / IDCG@k
```

Binary relevance (1/0). nDCG is sensitive to k; always report it with k.
Statistically unreliable on very small datasets (< 20 examples).

## Generation Metrics

### Exact Match

`rag/evaluation/metrics/exact_match.py`

```
EM = (number of exact matches) / (number of examples)
```

Default normalisation: case-insensitive, whitespace collapsed. Strict; best
used for short factoid answers. Not suitable for long-form output.

Configuration: `case_sensitive`, `strip_punctuation`.

### Token F1

`rag/evaluation/metrics/token_f1.py`

```
Precision = |gen_tokens ∩ exp_tokens| / |gen_tokens|
Recall    = |gen_tokens ∩ exp_tokens| / |exp_tokens|
F1        = 2 × Precision × Recall / (Precision + Recall)
```

Tokens are multisets. Tokenisation is a lowercase word-boundary regex.
Rewards overlap, not semantic equivalence. Combine with Exact Match.

## Performance Metrics

### Latency

`rag/evaluation/metrics/latency.py`

Mean wall-clock time for each pipeline stage in milliseconds. Stages:
`retrieval_ms`, `reranking_ms`, `prompt_construction_ms`, `generation_ms`,
`end_to_end_ms`. Single-run values are noisy; use the benchmarking layer
for distributions.

### Throughput

`rag/evaluation/metrics/throughput.py`

```
Throughput (examples/s) = n_examples / total_duration_s
```

Single-process, single-threaded; does not predict concurrent capacity.

### Memory Usage

`rag/evaluation/metrics/memory_usage.py`

Peak RSS in megabytes across collected `ResourceSnapshot` objects. Requires
`capture_resources=True` and psutil. Peak is approximated from discrete
snapshots; true peak may be higher.

## Adding a Custom Metric

```python
from rag.evaluation.metrics.base import BaseMetric
from rag.evaluation.types import EvaluationPrediction, MetricResult

class MyMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "my_metric"

    def evaluate(self, predictions):
        value = ...  # your computation
        return MetricResult(metric_name=self.name, value=value)

runner = EvaluationRunner(config, retriever=r, extra_metrics=[MyMetric()])
```
