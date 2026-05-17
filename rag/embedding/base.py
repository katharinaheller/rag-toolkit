from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple

VALID_RETRIEVAL_MODES: frozenset = frozenset({"dense", "sparse", "hybrid"})


class BaseEmbedder(ABC):
    """Contract for all embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def embed_queries(self, texts: List[str]) -> List[List[float]]:
        ...

    @abstractmethod
    def dimension(self) -> Optional[int]:
        """Native dense dimension, or None if not applicable."""

    def embed_documents_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sparse embeddings.")

    def embed_queries_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support sparse embeddings.")

    def embed_documents_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        return self.embed_documents(texts), self.embed_documents_sparse(texts)

    def embed_queries_hybrid(self, texts: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        return self.embed_queries(texts), self.embed_queries_sparse(texts)

    def supported_modes(self) -> Set[str]:
        return {"dense"}

    def default_retrieval_mode(self) -> str:
        return "dense"

    def is_mrl_model(self) -> bool:
        return False

    def model_type(self) -> str:
        return self.__class__.__name__.lower()

    def validate_embedding_contract(self) -> None:
        """Check that declared capabilities are internally consistent."""
        modes = self.supported_modes()
        dim = self.dimension()
        if "dense" in modes and dim is None:
            raise ValueError(
                f"{self.model_type()} declares dense support but returns dimension=None."
            )

    def validate_projection_config(
        self,
        target_dim: Optional[int],
        resolved_method: str,
        original_method: str,
        model_aware: bool,
    ) -> None:
        ...
