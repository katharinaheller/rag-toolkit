DOCUMENT_ID: doc_scale_017
TITLE: BM25 in Large-Scale Information Retrieval
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v7

CONTENT:
BM25 serves as the standard first-stage retrieval model in large-scale
commercial search engines and is the basis of the Elasticsearch and Apache
Lucene relevance scoring implementations. At web scale the inverted index
spans billions of terms and trillions of document-term pairs, requiring
distributed index partitioning across hundreds of servers. Query processing
at scale uses techniques such as MaxScore and WAND (Weak AND) to skip
low-scoring documents during score accumulation, maintaining sub-millisecond
query latency even for long-tail queries. BM25 scores are typically used to
select a candidate set of thousands of documents that are then re-ranked
by more expensive neural models, a two-stage retrieve-then-rerank pipeline
that balances scalability with ranking quality.
