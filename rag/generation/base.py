from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag.generation.models import GenerationResult


class BaseGenerationStrategy(ABC):
    """Abstract base for RAG generation strategies.

    Contract:
        - context_chunks contains raw text strings only.
        - generate() must not raise; errors are surfaced via GenerationResult.error.
        - No side effects.
    """

    @abstractmethod
    def generate(self, query: str, context_chunks: List[str]) -> GenerationResult:
        """Generate an answer given a query and retrieved context chunks."""
        raise NotImplementedError
