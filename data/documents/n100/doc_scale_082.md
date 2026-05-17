DOCUMENT_ID: doc_scale_082
TITLE: Mean Reciprocal Rank (MRR)
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v2

CONTENT:
Mean Reciprocal Rank (MRR) measures the average position of the first relevant
document across a set of queries. For a single query, the Reciprocal Rank (RR)
is 1 / rank_r, where rank_r is the position of the first relevant document
in the ranked result list. If no relevant document appears in the top-k,
the RR for that query is 0. The MRR is the arithmetic mean of RR over all
queries in the evaluation set. MRR is particularly appropriate for tasks
where only a single relevant document needs to be found, such as navigational
queries or direct question answering where a single passage contains the
answer. MRR values above 0.7 are generally considered good for production
retrieval systems; values below 0.5 indicate that the retriever frequently
misses the most relevant passage in the top positions.
