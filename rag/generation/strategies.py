from __future__ import annotations

import datetime
import logging
import time
from typing import List, Optional, Tuple, Union

from rag.generation.base import BaseGenerationStrategy
from rag.generation.client import OllamaClient
from rag.generation.context import ContextPreparer, _ContextItem
from rag.generation.exceptions import GenerationError
from rag.generation.models import GenerationResult
from rag.generation.prompt_builder import (
    REFINE_INITIAL_TEMPLATE,
    REFINE_UPDATE_TEMPLATE,
    PromptBuilder,
)

logger = logging.getLogger(__name__)


def _utc_timestamp() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _build_result(
    client: OllamaClient,
    builder: PromptBuilder,
    answer: str,
    prompt: str,
    context_chars: int,
    latency_ms: float,
    timestamp: str,
    raw_response: Optional[dict],
    error: Optional[str],
    strategy_name: str,
) -> GenerationResult:
    """Shared result construction used by all strategies."""
    return GenerationResult(
        answer=answer,
        prompt=prompt,
        model=client._config.model_name,
        strategy=strategy_name,
        template_name=builder.template.name,
        template_version=builder.template.version,
        latency_ms=round(latency_ms, 2),
        prompt_chars=len(prompt),
        context_chars=context_chars,
        inference_parameters=dict(client._config.ollama_options),
        timestamp=timestamp,
        raw_response=raw_response,
        error=error,
    )


class SimpleRAGStrategy(BaseGenerationStrategy):
    """Single-pass RAG: concatenate context, one prompt, one LLM call."""

    def __init__(
        self,
        client: OllamaClient,
        prompt_builder: PromptBuilder,
        context_preparer: ContextPreparer,
    ) -> None:
        self._client = client
        self._prompt_builder = prompt_builder
        self._context_preparer = context_preparer

    def generate(self, query: str, context_chunks: List[Union[str, _ContextItem]]) -> GenerationResult:
        timestamp = _utc_timestamp()
        t0 = time.perf_counter()

        clean_chunks = self._context_preparer.prepare(context_chunks)
        context_chars = sum(len(c) for c in clean_chunks)
        prompt = self._prompt_builder.build(query, clean_chunks)

        logger.debug(
            "SimpleRAGStrategy: %d chunks → %d context chars, %d prompt chars.",
            len(clean_chunks), context_chars, len(prompt),
        )

        try:
            raw_response = self._client.generate(prompt)
            latency_ms = (time.perf_counter() - t0) * 1_000
            return _build_result(
                self._client, self._prompt_builder,
                answer=str(raw_response.get("response", "")),
                prompt=prompt, context_chars=context_chars,
                latency_ms=latency_ms, timestamp=timestamp,
                raw_response=dict(raw_response), error=None,
                strategy_name=self.__class__.__name__,
            )

        except GenerationError as exc:
            latency_ms = (time.perf_counter() - t0) * 1_000
            logger.error("SimpleRAGStrategy failed after %.0f ms: %s", latency_ms, exc.message)
            return _build_result(
                self._client, self._prompt_builder,
                answer="", prompt=prompt, context_chars=context_chars,
                latency_ms=latency_ms, timestamp=timestamp,
                raw_response=None, error=str(exc),
                strategy_name=self.__class__.__name__,
            )

        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1_000
            logger.exception("SimpleRAGStrategy unexpected error after %.0f ms.", latency_ms)
            return _build_result(
                self._client, self._prompt_builder,
                answer="", prompt=prompt, context_chars=context_chars,
                latency_ms=latency_ms, timestamp=timestamp,
                raw_response=None, error=f"Unexpected error: {exc}",
                strategy_name=self.__class__.__name__,
            )


class RefineRAGStrategy(BaseGenerationStrategy):
    """Multi-pass iterative refinement.

    Initial answer from the first chunk, then one LLM call per remaining chunk.
    Total LLM calls = len(context_chunks).
    """

    def __init__(self, client: OllamaClient, context_preparer: ContextPreparer) -> None:
        self._client = client
        self._context_preparer = context_preparer
        self._initial_builder = PromptBuilder(REFINE_INITIAL_TEMPLATE)
        self._update_builder = PromptBuilder(REFINE_UPDATE_TEMPLATE)

    def generate(self, query: str, context_chunks: List[Union[str, _ContextItem]]) -> GenerationResult:
        timestamp = _utc_timestamp()
        t0 = time.perf_counter()

        clean_chunks = self._context_preparer.prepare(context_chunks)
        context_chars = sum(len(c) for c in clean_chunks)

        if not clean_chunks:
            return self._empty_context_result(query, timestamp, t0)

        current_answer, prompt, raw_response, error = self._initial_pass(query, clean_chunks[0])
        if error is not None:
            latency_ms = (time.perf_counter() - t0) * 1_000
            return self._error_result(prompt, context_chars, timestamp, latency_ms, error)

        for chunk in clean_chunks[1:]:
            current_answer, prompt, raw_response, error = self._refine_pass(current_answer, chunk)
            if error is not None:
                latency_ms = (time.perf_counter() - t0) * 1_000
                return self._error_result(prompt, context_chars, timestamp, latency_ms, error)

        latency_ms = (time.perf_counter() - t0) * 1_000
        return _build_result(
            self._client, self._update_builder,
            answer=current_answer, prompt=prompt, context_chars=context_chars,
            latency_ms=latency_ms, timestamp=timestamp,
            raw_response=dict(raw_response) if raw_response else None,
            error=None, strategy_name=self.__class__.__name__,
        )

    def _initial_pass(self, query: str, first_chunk: str) -> Tuple:
        prompt = self._initial_builder.build(query, [first_chunk])
        try:
            raw = self._client.generate(prompt)
            return str(raw.get("response", "")), prompt, dict(raw), None
        except GenerationError as exc:
            logger.error("RefineRAGStrategy initial pass failed: %s", exc.message)
            return "", prompt, None, str(exc)
        except Exception as exc:
            logger.exception("RefineRAGStrategy initial pass unexpected error.")
            return "", prompt, None, f"Unexpected error: {exc}"

    def _refine_pass(self, current_answer: str, new_chunk: str) -> Tuple:
        # In REFINE_UPDATE_TEMPLATE: {context} = existing answer, {query} = new chunk.
        prompt = self._update_builder.build(query=new_chunk, context_chunks=[current_answer])
        try:
            raw = self._client.generate(prompt)
            return str(raw.get("response", "")), prompt, dict(raw), None
        except GenerationError as exc:
            logger.error("RefineRAGStrategy refine pass failed: %s", exc.message)
            return current_answer, prompt, None, str(exc)
        except Exception as exc:
            logger.exception("RefineRAGStrategy refine pass unexpected error.")
            return current_answer, prompt, None, f"Unexpected error: {exc}"

    def _empty_context_result(self, query: str, timestamp: str, t0: float) -> GenerationResult:
        prompt = self._initial_builder.build(query, [])
        latency_ms = (time.perf_counter() - t0) * 1_000
        return _build_result(
            self._client, self._initial_builder,
            answer="", prompt=prompt, context_chars=0,
            latency_ms=latency_ms, timestamp=timestamp,
            raw_response=None, error="No valid context chunks provided after sanitisation.",
            strategy_name=self.__class__.__name__,
        )

    def _error_result(
        self, prompt: str, context_chars: int, timestamp: str, latency_ms: float, error: str
    ) -> GenerationResult:
        return _build_result(
            self._client, self._update_builder,
            answer="", prompt=prompt, context_chars=context_chars,
            latency_ms=latency_ms, timestamp=timestamp,
            raw_response=None, error=error,
            strategy_name=self.__class__.__name__,
        )
