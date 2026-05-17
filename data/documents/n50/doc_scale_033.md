DOCUMENT_ID: doc_scale_033
TITLE: FAISS HNSW: Hierarchical Navigable Small World
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v3

CONTENT:
HNSW (Hierarchical Navigable Small World) is a graph-based approximate
nearest-neighbour index that achieves O(log n) query time through a
multi-layer proximity graph structure. Each layer of the graph contains a
subset of vectors connected to their approximate nearest neighbours with
long-range edges at higher layers enabling fast coarse navigation and
short-range edges at the lowest layer enabling fine-grained search.
In FAISS, the HNSW index is implemented as IndexHNSWFlat, which stores the
full vectors without quantisation alongside the graph structure. The M
parameter controls the number of neighbours per node (typically 16 to 64),
trading off between build time, memory, and recall: higher M yields better
recall at the cost of more memory. HNSW is often preferred for dynamic
corpora because nodes can be inserted incrementally without rebuilding
the full index.
