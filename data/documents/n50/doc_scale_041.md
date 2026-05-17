DOCUMENT_ID: doc_scale_041
TITLE: Hybrid Retrieval: Combining Sparse and Dense
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v1

CONTENT:
Hybrid retrieval combines sparse lexical retrieval (BM25) with dense semantic
retrieval (bi-encoder models) to leverage the complementary strengths of each
approach. Sparse retrieval excels at exact keyword matching and rare entity
retrieval, while dense retrieval handles synonym matching, paraphrase, and
complex semantic similarity. The output of each retrieval method is a ranked
list of documents with associated scores, and a fusion strategy combines these
lists into a single merged ranking. Empirical evaluations consistently show
that hybrid retrieval outperforms either pure sparse or pure dense retrieval
on most benchmarks, including the MS MARCO passage and document ranking tasks.
The gain is largest for queries that combine keyword specificity with semantic
complexity, a common pattern in enterprise search over technical documentation.
