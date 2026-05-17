from rag.indexing.base import BaseDenseIndex, BaseSparseIndex
from rag.indexing.config import IndexConfig
from rag.indexing.backends.brute_force import BruteForceIndex
from rag.indexing.sparse.sparse_index import SparseIndex

try:
    from rag.indexing.backends.faiss_index import FAISSIndex
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


def create_dense_index(config: IndexConfig) -> BaseDenseIndex:
    """Instantiate the configured dense index backend."""
    index_dir = config.index_dir / "dense"

    if config.dense.backend == "brute_force":
        return BruteForceIndex(config.dense, index_dir)

    if config.dense.backend == "faiss":
        if not _FAISS_AVAILABLE:
            raise ImportError(
                "FAISS backend requested but not installed. Install: pip install faiss-cpu"
            )
        return FAISSIndex(config.dense, index_dir)

    raise ValueError(f"Unknown dense backend: '{config.dense.backend}'. Valid: 'brute_force', 'faiss'.")


def create_sparse_index(config: IndexConfig) -> BaseSparseIndex:
    """Instantiate the sparse lexical index."""
    return SparseIndex(config.sparse, config.index_dir / "sparse")
