DOCUMENT_ID: doc_scale_088
TITLE: End-to-End vs Component-Level Evaluation
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v8

CONTENT:
RAG system evaluation can be conducted at the component level (retrieval
metrics, generation metrics separately) or end-to-end (measuring the quality
of the final generated answer given the full pipeline). End-to-end metrics
are the ultimate measure of user-facing quality but mask the source of errors:
a low end-to-end Token F1 could result from poor retrieval, poor generation,
or both. Component-level evaluation pinpoints the bottleneck by measuring
retrieval quality independently of generation quality, enabling targeted
improvements. A recommended evaluation protocol runs both: first evaluate
retrieval in isolation using nDCG@k and MRR, then evaluate generation in
isolation using pre-provided ground-truth contexts (bypassing retrieval), and
finally evaluate end-to-end to measure the compounding effect of both
components. Comparing end-to-end quality with oracle retrieval quality
reveals the ceiling imposed by generation quality on the current system.
