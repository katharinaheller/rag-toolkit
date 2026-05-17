DOCUMENT_ID: doc_scale_089
TITLE: Recall@k and Hit Rate
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v9

CONTENT:
Recall@k (also written R@k) measures the fraction of queries for which at
least one relevant document appears in the top-k retrieved results. Unlike
Context Recall (which measures the fraction of all relevant documents
retrieved), Recall@k is a binary measure per query: either the relevant
document is in the top-k (1) or it is not (0), and the metric is the mean
over all queries. For RAG evaluation where each query typically has a single
ground-truth relevant document, Recall@k and Hit Rate are equivalent.
Recall@1 measures whether the most relevant document is retrieved in the first
position. Recall@5 and Recall@10 are commonly reported, as they align with
typical top-k values used in RAG pipelines. High Recall@10 ensures that the
relevant passage is almost always available for the generator, while the
precision of the full top-10 set affects how much noise the generator must
filter out.
