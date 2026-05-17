DOCUMENT_ID: doc_scale_099
TITLE: Corpus Scaling and Index Rebuild Strategies
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v9

CONTENT:
As the document corpus grows, the retrieval index must be updated to reflect
new and modified documents. Two update strategies are common: incremental
indexing and full rebuild. Incremental indexing appends new vectors to the
index without rebuilding it from scratch, which is fast and avoids downtime
but may produce suboptimal index structure for IVF-based indexes as the
cluster centroids drift from the actual data distribution. Full rebuild
re-trains the index from scratch on all vectors, producing optimal index
structure at the cost of a longer build time and temporary downtime during
the rebuild. For HNSW indexes, incremental insertion is always safe because
HNSW does not require training. For IVF indexes, periodic full rebuilds
(triggered when the number of new documents exceeds a threshold such as 10%
of the existing corpus) restore optimal cluster balance.
