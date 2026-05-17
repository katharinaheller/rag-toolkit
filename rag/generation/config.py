from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable configuration for a local Ollama generation call."""

    model_name: str

    base_url: str = "http://ollama:11434"
    endpoint: str = "/api/generate"

    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 42
    top_p: float = 1.0
    repeat_penalty: float = 1.1

    timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 2.0

    max_context_chars: int = 4_000

    def __post_init__(self) -> None:
        if not self.model_name or not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if not self.base_url or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string.")
        if not self.endpoint or not self.endpoint.strip():
            raise ValueError("endpoint must be a non-empty string.")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0]; got {self.temperature}.")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1; got {self.max_tokens}.")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0.0, 1.0]; got {self.top_p}.")
        if self.repeat_penalty <= 0.0:
            raise ValueError(f"repeat_penalty must be > 0.0; got {self.repeat_penalty}.")
        if self.timeout <= 0.0:
            raise ValueError(f"timeout must be > 0.0; got {self.timeout}.")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0; got {self.max_retries}.")
        if self.retry_delay < 0.0:
            raise ValueError(f"retry_delay must be >= 0.0; got {self.retry_delay}.")
        if self.max_context_chars < 1:
            raise ValueError(f"max_context_chars must be >= 1; got {self.max_context_chars}.")

    @property
    def ollama_options(self) -> dict:
        """Ollama options sub-dict for the request body."""
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "seed": self.seed,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }

    @property
    def url(self) -> str:
        """Fully resolved Ollama endpoint URL."""
        return f"{self.base_url.rstrip('/')}{self.endpoint}"
