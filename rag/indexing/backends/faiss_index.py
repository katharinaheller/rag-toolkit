import json
from pathlib import Path
from typing import List, Optional

from rag.indexing.base import BaseDenseIndex
from rag.indexing.config import DenseIndexConfig
from rag.indexing.types import DenseCandidateResult, DenseIndexEntry, DenseQuery
from rag.indexing.utils import l2_normalize, validate_real_vector

try:
    import faiss
    import numpy as np
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class FAISSIndex(BaseDenseIndex):
    """Approximate nearest-neighbor dense search via FAISS."""

    def __init__(self, config: DenseIndexConfig, index_dir: Path) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError("FAISS not available. Install: pip install faiss-cpu")

        self._config = config
        self._index_dir = index_dir
        index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = index_dir / "faiss.index"
        self._id_map_path = index_dir / "faiss_id_map.json"
        self._id_map: List[str] = []
        self._faiss_index = None
        self._dim: Optional[int] = None

    def add(self, entries: List[DenseIndexEntry]) -> None:
        if not entries:
            return

        vectors = [e["dense_vector"] for e in entries]
        dim = len(vectors[0])

        if dim == 0:
            raise ValueError("Dense vectors must not be empty.")
        if any(len(v) != dim for v in vectors):
            raise ValueError("Inconsistent vector dimensions within the batch.")

        for vector in vectors:
            validate_real_vector(vector, "dense_vector")

        if self._config.dimension is not None and dim != self._config.dimension:
            raise ValueError(
                f"Vector dimension {dim} does not match config.dimension={self._config.dimension}."
            )

        if self._dim is None:
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(f"Dimension mismatch: index dim={self._dim}, new batch dim={dim}.")

        float_vectors = [[float(x) for x in v] for v in vectors]

        if self._config.metric == "cosine":
            float_vectors = [l2_normalize(v) for v in float_vectors]

        if self._faiss_index is None:
            self._faiss_index = self._build_index(dim)

        if (
            self._config.faiss.index_type == "ivf"
            and hasattr(self._faiss_index, "is_trained")
            and not self._faiss_index.is_trained
        ):
            min_size = self._config.faiss.ivf_nlist * 10
            if len(float_vectors) < min_size:
                raise ValueError(
                    f"FAISS IVF requires >= {min_size} training vectors "
                    f"(ivf_nlist={self._config.faiss.ivf_nlist}), got {len(float_vectors)}."
                )

        arr = np.array(float_vectors, dtype="float32")

        if hasattr(self._faiss_index, "is_trained") and not self._faiss_index.is_trained:
            self._faiss_index.train(arr)

        self._faiss_index.add(arr)
        self._id_map.extend(e["id"] for e in entries)

    def query(self, query: DenseQuery, k: int) -> List[DenseCandidateResult]:
        if k <= 0:
            return []
        if self._faiss_index is None or not self._id_map:
            return []
        if not isinstance(query, list):
            raise TypeError(f"FAISSIndex.query expects List[float], got {type(query).__name__}.")
        if not query:
            return []

        validate_real_vector(query, "query")
        q_vec = [float(x) for x in query]

        if self._dim is not None and len(q_vec) != self._dim:
            raise ValueError(f"Query dim {len(q_vec)} != index dim {self._dim}.")

        if self._config.metric == "cosine":
            q_vec = l2_normalize(q_vec)

        q = np.array([q_vec], dtype="float32")
        search_k = max(k * self._config.faiss.search_k_factor, k)
        scores, indices = self._faiss_index.search(q, search_k)

        results: List[DenseCandidateResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue

            final_score = float(score)
            if self._config.metric == "l2":
                final_score = -final_score

            results.append({"id": self._id_map[int(idx)], "score": final_score})

            if len(results) >= k:
                break

        results.sort(key=lambda r: (-r["score"], r["id"]))
        return results

    def persist(self) -> None:
        if self._faiss_index is None:
            return
        faiss.write_index(self._faiss_index, str(self._index_path))
        with self._id_map_path.open("w", encoding="utf-8") as f:
            json.dump(self._id_map, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if self._index_path.exists():
            self._faiss_index = faiss.read_index(str(self._index_path))
            self._dim = self._faiss_index.d

        if self._id_map_path.exists():
            with self._id_map_path.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
                if not isinstance(loaded, list) or any(not isinstance(i, str) for i in loaded):
                    raise ValueError("FAISS id map must be a JSON list of strings.")
                self._id_map = loaded

    def _build_index(self, dim: int):
        metric = (
            faiss.METRIC_INNER_PRODUCT
            if self._config.metric in {"cosine", "dot"}
            else faiss.METRIC_L2
        )
        cfg = self._config.faiss

        if cfg.index_type == "flat":
            return (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )

        if cfg.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, cfg.hnsw_m, metric)
            idx.hnsw.efConstruction = cfg.hnsw_ef_construction
            return idx

        if cfg.index_type == "ivf":
            quantizer = (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
            return faiss.IndexIVFFlat(quantizer, dim, cfg.ivf_nlist, metric)

        raise ValueError(f"Unknown FAISS index type: '{cfg.index_type}'. Valid: 'flat', 'hnsw', 'ivf'.")
