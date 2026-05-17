DOCUMENT_ID: doc_scale_014
TITLE: BM25 Limitations: Vocabulary Mismatch
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v4

CONTENT:
The fundamental limitation of BM25 and lexical retrieval in general is
vocabulary mismatch: the model can only match documents that share exact
token overlap with the query. This means that a query asking about
"automobile engines" will not retrieve documents that exclusively use the
term "car motors" even if the content is semantically identical. Synonymy,
paraphrase, abbreviation expansion, and cross-lingual retrieval are all
beyond BM25's capability without external preprocessing. This limitation
motivated the development of dense retrieval models that map both queries
and documents into a shared semantic embedding space, allowing soft matches
across vocabulary boundaries. In practice, hybrid retrieval systems that
combine BM25 with dense retrieval achieve better overall performance than
either approach alone, capturing the complementary strengths of exact-match
precision and semantic recall.
