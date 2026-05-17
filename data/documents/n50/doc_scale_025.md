DOCUMENT_ID: doc_scale_025
TITLE: Approximate Nearest Neighbour Search for Dense Retrieval
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v5

CONTENT:
Exact nearest-neighbour search over a corpus of millions of embedding vectors
requires computing the dot product of the query vector with every document
vector, which scales linearly with corpus size. Approximate nearest-neighbour
(ANN) algorithms sacrifice a small amount of recall for dramatically improved
query latency. The most widely used ANN methods include HNSW (Hierarchical
Navigable Small World), which builds a multi-layered graph of approximate
neighbourhood links and achieves O(log n) query time; IVF (Inverted File),
which quantises vectors into cluster centroids and searches only the nearest
clusters at query time; and Product Quantisation (PQ), which compresses each
vector into a compact code, reducing memory by 8-32x at the cost of some
recall. Libraries such as FAISS, ScaNN, and Annoy implement these algorithms
and expose a unified interface that makes swapping between exact and
approximate search straightforward.
