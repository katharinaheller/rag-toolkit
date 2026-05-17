DOCUMENT_ID: doc_scale_048
TITLE: Hybrid Retrieval in Production Systems
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v8

CONTENT:
Deploying a hybrid retrieval system in production requires maintaining two
parallel index infrastructure components: an inverted index for BM25 (or
learned sparse) retrieval and a dense vector index for semantic retrieval.
These indexes must be kept in synchrony when documents are added, updated,
or deleted. Serving architecture typically routes incoming queries to both
indexes in parallel (latency equals the slower of the two), then applies the
fusion step before returning top-k results to the caller. Cache warming
strategies can pre-populate query result caches for frequently-issued queries,
reducing the fraction of requests that hit both indexes. Monitoring should
separately track sparse recall, dense recall, and hybrid recall on a
continuous evaluation set to detect index drift or model degradation
independently for each retrieval component.
