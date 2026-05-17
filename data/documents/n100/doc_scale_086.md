DOCUMENT_ID: doc_scale_086
TITLE: Throughput and Queries Per Second
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v6

CONTENT:
Throughput, measured in queries per second (QPS) or examples per second,
quantifies the rate at which a RAG system can process requests under
sustained load. Unlike latency (which measures a single request's response
time), throughput measures system capacity. The relationship between latency
and throughput depends on concurrency: at low concurrency throughput scales
linearly with added workers, but at high concurrency resource contention
(GPU saturation, memory bandwidth limits) causes latency to increase while
throughput plateaus. For retrieval-only benchmarks, measuring throughput in
a multi-threaded concurrent benchmark reveals the index's saturation point.
For end-to-end RAG, LLM generation typically dominates and single-GPU
throughput is constrained by the model's token generation rate. Horizontal
scaling with multiple LLM replicas is the standard approach to increase
end-to-end RAG throughput beyond single-GPU limits.
