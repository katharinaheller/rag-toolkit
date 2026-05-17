from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class GenerationResult:
    """Immutable record of a single generation call.

    `error is None` indicates success. `answer` is "" on failure, never None.
    """

    answer: str
    prompt: str
    model: str
    strategy: str
    template_name: str
    template_version: str

    latency_ms: float = 0.0
    prompt_chars: int = 0
    context_chars: int = 0

    inference_parameters: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    raw_response: Optional[Mapping[str, Any]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and bool(self.answer)

    def to_dict(self) -> dict:
        """Flat dictionary suitable for logging or serialization."""
        return {
            "answer": self.answer,
            "prompt": self.prompt,
            "model": self.model,
            "strategy": self.strategy,
            "template_name": self.template_name,
            "template_version": self.template_version,
            "latency_ms": self.latency_ms,
            "prompt_chars": self.prompt_chars,
            "context_chars": self.context_chars,
            "inference_parameters": dict(self.inference_parameters),
            "timestamp": self.timestamp,
            "error": self.error,
            "success": self.success,
        }
