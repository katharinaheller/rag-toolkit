from __future__ import annotations

import logging
import time
from typing import Any, Mapping

import requests

from rag.generation.config import GenerationConfig
from rag.generation.exceptions import LLMConnectionError, LLMResponseError, LLMTimeoutError

logger = logging.getLogger(__name__)


class OllamaClient:
    """Synchronous HTTP client for the Ollama /api/generate endpoint.

    Retries with fixed delay on transient errors (connection, timeout).
    Response errors (4xx/5xx, bad JSON) are not retried — they're deterministic.
    Not thread-safe.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self._config = config

    def generate(self, prompt: str) -> Mapping[str, Any]:
        """Send a generate request and return the decoded JSON response."""
        payload = self._build_payload(prompt)
        last_exception: Exception | None = None
        total = self._config.max_retries + 1

        for attempt in range(1, total + 1):
            try:
                return self._execute(payload, attempt, total)
            except (LLMConnectionError, LLMTimeoutError) as exc:
                last_exception = exc
                if attempt < total:
                    logger.warning(
                        "Attempt %d/%d failed (%s); retrying in %.1f s.",
                        attempt, total, exc.message, self._config.retry_delay,
                    )
                    time.sleep(self._config.retry_delay)
            except LLMResponseError:
                raise

        assert last_exception is not None
        raise last_exception

    def _build_payload(self, prompt: str) -> dict:
        return {
            "model": self._config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": dict(self._config.ollama_options),
        }

    def _execute(self, payload: dict, attempt: int, total: int) -> Mapping[str, Any]:
        logger.debug(
            "OllamaClient attempt %d/%d → %s (model=%s, timeout=%.1fs).",
            attempt, total, self._config.url, self._config.model_name, self._config.timeout,
        )

        try:
            http_response = requests.post(
                self._config.url, json=payload, timeout=self._config.timeout
            )
        except requests.exceptions.Timeout as exc:
            raise LLMTimeoutError(
                f"Request to {self._config.url!r} timed out after {self._config.timeout:.1f}s."
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self._config.url!r}. "
                f"Check base_url (got {self._config.base_url!r})."
            ) from exc

        if not http_response.ok:
            raise LLMResponseError(
                f"Ollama returned HTTP {http_response.status_code} for {self._config.url!r}. "
                f"Body: {http_response.text[:500]!r}"
            )

        try:
            body: dict = http_response.json()
        except ValueError as exc:
            raise LLMResponseError(
                f"Ollama response is not valid JSON. Body: {http_response.text[:500]!r}"
            ) from exc

        if "error" in body:
            raise LLMResponseError(f"Ollama error: {body['error']!r}")

        if "response" not in body:
            raise LLMResponseError(
                f"Ollama response missing 'response' field. Keys: {list(body.keys())}"
            )

        logger.debug("OllamaClient: received %d chars.", len(body["response"]))
        return body
