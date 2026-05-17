DOCUMENT_ID: doc_scale_038
TITLE: Choosing a FAISS Index for RAG Pipelines
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v8

CONTENT:
Selecting the appropriate FAISS index type for a RAG pipeline depends on
three factors: corpus size, memory budget, and recall-latency requirements.
For corpora up to a few hundred thousand vectors and unlimited memory,
IndexFlatIP provides exact search with zero configuration. For corpora of
one to ten million vectors with moderate memory constraints, IndexHNSWFlat
offers high recall (>99%) with sub-millisecond query latency and incremental
insertion support. For corpora of tens of millions of vectors with strict
memory limits, IndexIVFPQ provides the best compression ratio at some cost
to recall. For hundred-million to billion-scale corpora, HNSW combined with
scalar quantisation (IVFHNSW or IVFSQ) achieves the best balance of speed,
recall, and memory. In practice, starting with IndexFlatIP for correctness
validation and then migrating to HNSW or IVF as corpus size grows is a
reliable engineering pattern.
