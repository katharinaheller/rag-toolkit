DOCUMENT_ID: doc_scale_040
TITLE: Vector Databases Beyond FAISS
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v10

CONTENT:
Beyond FAISS, a growing ecosystem of managed vector databases has emerged
that provides production-ready similarity search with additional features
such as metadata filtering, multi-tenant isolation, and horizontal scaling.
Weaviate, Pinecone, Milvus, Qdrant, and Chroma are among the most widely
used options. These systems typically wrap HNSW or IVF internally and add
a query layer that supports filtered similarity search — for example,
retrieving the top-k vectors among those satisfying a metadata predicate.
Filtering can be applied as pre-filtering (restricting the search space
before ANN search), post-filtering (filtering the top-k results after
search), or hybrid single-stage filtering integrated into the ANN algorithm.
The choice between self-hosted FAISS and a managed vector database involves
trading operational simplicity for control, cost, and feature richness.
