DOCUMENT_ID: doc_scale_085
TITLE: Latency Metrics for RAG Systems
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v5

CONTENT:
Latency measurement in RAG systems should decompose end-to-end latency into
its constituent stages: query encoding, index search, context preparation,
and LLM generation. Each stage's latency distribution should be characterised
by mean, median, p95, and p99 percentiles, since the mean alone is misleading
when the distribution has a heavy tail. The median latency is more robust than
the mean for non-Gaussian distributions, and p95 latency represents the worst-
case experience for 95% of users. Benchmark runs should include warmup
iterations to exclude one-time model loading overhead from the reported
statistics. At the system level, end-to-end latency should be measured under
realistic concurrent load rather than sequentially, because contention for
GPU memory and compute resources can increase tail latency substantially
beyond the single-request median.
