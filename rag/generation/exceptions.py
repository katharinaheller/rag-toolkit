from __future__ import annotations


class GenerationError(Exception):
    """Base class for all rag.generation errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.message!r})"


class LLMConnectionError(GenerationError):
    """Raised when the HTTP client cannot connect to Ollama."""


class LLMTimeoutError(GenerationError):
    """Raised when an Ollama request exceeds the configured timeout."""


class LLMResponseError(GenerationError):
    """Raised when Ollama responds with an invalid or error payload."""
