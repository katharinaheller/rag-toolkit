DOCUMENT_ID: doc_scale_046
TITLE: Re-Ranking in Hybrid Retrieval Pipelines
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v6

CONTENT:
Re-ranking is a post-retrieval stage that applies a more expensive model to
the short-listed candidate set produced by hybrid first-stage retrieval.
Cross-encoder re-rankers such as MonoT5 and MS MARCO-trained BERT cross-
encoders score each query-passage pair jointly, allowing full attention across
both inputs. Because re-ranking operates on a small candidate set (typically
100-1000 documents) rather than the full corpus, the latency overhead is
acceptable even for slow cross-encoders. Re-ranking substantially improves
nDCG@10 compared to first-stage retrieval alone, often by 5-10 points
on benchmarks such as MS MARCO. The practical challenge is that re-ranking
adds a second model loading and inference step to the retrieval pipeline,
increasing memory and latency. Distilled cross-encoder models that are smaller
and faster than full BERT-scale models are commonly used to balance quality
and efficiency.
