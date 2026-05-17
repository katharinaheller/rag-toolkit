DOCUMENT_ID: doc_scale_037
TITLE: FAISS Index Persistence and Serialisation
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v7

CONTENT:
FAISS provides built-in serialisation for all index types through the
write_index() and read_index() functions, which save and load indexes to
binary files on disk. A flat index storing one million 1024-dimensional
float32 vectors occupies approximately 4 GB on disk, while an IVF-PQ
compressed index may occupy as little as 64 MB for the same collection.
For RAG systems that need fast startup times, the index should be persisted
after the initial build and loaded from disk on subsequent starts rather than
being rebuilt from scratch. Incremental addition of new vectors is supported
through the add() method; however, IVF indexes require periodic rebalancing
(re-clustering) when the distribution of new vectors differs significantly
from the training data. FAISS indexes are not thread-safe by default; external
locking must be applied when multiple threads call search() concurrently.
