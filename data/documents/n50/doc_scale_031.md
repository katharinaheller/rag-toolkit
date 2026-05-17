DOCUMENT_ID: doc_scale_031
TITLE: FAISS: Facebook AI Similarity Search
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v1

CONTENT:
FAISS (Facebook AI Similarity Search) is an open-source library developed by
Meta AI Research for efficient similarity search and clustering of dense
vectors. Originally released in 2017, FAISS provides a suite of index types
that trade off between memory usage, build time, and query accuracy. The
library is implemented in C++ with Python bindings and supports both CPU and
GPU execution. FAISS is widely used as the vector index backend in RAG
pipelines, recommendation systems, and image search applications.
The library implements brute-force flat search (IndexFlatL2, IndexFlatIP),
quantised search (IndexPQ, IndexIVFPQ), graph-based approximate search
(IndexHNSWFlat), and composite indexes that combine multiple strategies.
GPU-accelerated FAISS indexes achieve query throughputs of millions of
vectors per second, making them suitable for real-time retrieval over
billion-scale vector collections.
