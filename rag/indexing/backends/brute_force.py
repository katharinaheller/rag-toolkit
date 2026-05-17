import json
from pathlib import Path
from typing import List, Optional, Tuple

from rag.indexing.base import BaseDenseIndex
from rag.indexing.config import DenseIndexConfig
from rag.indexing.types import DenseCandidateResult, DenseIndexEntry, DenseQuery
from rag.indexing.utils import cosine_similarity, dot_product, l2_distance, validate_real_vector


class BruteForceIndex(BaseDenseIndex):
    """Exact dense search via exhaustive pairwise comparison."""

    def __init__(self, config: DenseIndexConfig, index_dir: Path) -> None:
        self._config = config
        self._store_path = index_dir / "brute_force_entries.jsonl"
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[Tuple[str, List[float]]] = []
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
            raise ValueError(
                f"Dimension mismatch: index dim={self._dim}, new batch dim={dim}."
            )

        for e in entries:
            self._entries.append((e["id"], [float(x) for x in e["dense_vector"]]))

    def query(self, query: DenseQuery, k: int) -> List[DenseCandidateResult]:
        if k <= 0:
            return []
        if not isinstance(query, list):
            raise TypeError(f"BruteForceIndex.query expects List[float], got {type(query).__name__}.")
        if not query:
            return []

        validate_real_vector(query, "query")
        q = [float(x) for x in query]

        if self._dim is not None and len(q) != self._dim:
            raise ValueError(f"Query dim {len(q)} != index dim {self._dim}.")
        if not self._entries:
            return []

        scored: List[DenseCandidateResult] = [
            {"id": doc_id, "score": float(self._score(q, vector))}
            for doc_id, vector in self._entries
        ]
        scored.sort(key=lambda r: (-r["score"], r["id"]))
        return scored[:k]

    def persist(self) -> None:
        with self._store_path.open("w", encoding="utf-8") as f:
            for doc_id, vector in self._entries:
                f.write(json.dumps({"id": doc_id, "dense_vector": vector}) + "\n")

    def load(self) -> None:
        self._entries = []
        self._dim = None

        if not self._store_path.exists():
            return

        with self._store_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                vector = [float(x) for x in record["dense_vector"]]
                validate_real_vector(vector, "dense_vector")
                self._entries.append((record["id"], vector))

        if self._entries:
            self._dim = len(self._entries[0][1])

    def _score(self, a: List[float], b: List[float]) -> float:
        if self._config.metric == "cosine":
            return cosine_similarity(a, b)
        if self._config.metric == "dot":
            return dot_product(a, b)
        if self._config.metric == "l2":
            return -l2_distance(a, b)
        raise ValueError(f"Unsupported metric: '{self._config.metric}'.")
