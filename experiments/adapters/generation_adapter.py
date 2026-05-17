"""Adapter for ``rag.generation``.

Wraps :class:`SimpleRAGStrategy` behind a thin builder so generation suites can
be toggled off by configuration without touching their code. When generation
is disabled (or Ollama is unreachable), generation calls return a stub
:class:`GenerationResult`-like dict carrying ``error="generation_disabled"``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from rag.generation.client import OllamaClient
from rag.generation.config import GenerationConfig
from rag.generation.context import ContextPreparer
from rag.generation.prompt_builder import PromptBuilder, STRICT_RAG_TEMPLATE
from rag.generation.strategies import SimpleRAGStrategy
from rag.generation.models import GenerationResult

from experiments.configs.settings import SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class BuiltGenerator:
    """A configured generation pipeline plus an availability flag."""

    strategy: Optional[SimpleRAGStrategy]
    available: bool
    reason: str = ""
    model_name: str = ""

    def generate(self, query: str, contexts: List[str]):
        if not self.available or self.strategy is None:
            return _disabled_result(query, contexts, self.reason, self.model_name)
        return self.strategy.generate(query, contexts)


def _disabled_result(
    query: str, contexts: List[str], reason: str, model_name: str
) -> GenerationResult:
    """Build a GenerationResult representing a deliberately skipped call."""
    return GenerationResult(
        answer="",
        prompt="",
        model=model_name or "unknown",
        strategy="DisabledStrategy",
        template_name="none",
        template_version="0.0.0",
        latency_ms=0.0,
        prompt_chars=0,
        context_chars=sum(len(c) for c in contexts),
        inference_parameters={},
        timestamp="",
        raw_response=None,
        error=f"generation_disabled: {reason}",
    )


def _probe_ollama(url: str, timeout_s: float) -> tuple[bool, str]:
    """Cheap reachability probe so we fail fast and clearly."""
    try:
        import requests
        resp = requests.get(f"{url.rstrip('/')}/api/tags", timeout=min(timeout_s, 5.0))
        if resp.ok:
            return True, ""
        return False, f"ollama HTTP {resp.status_code}"
    except Exception as exc:
        return False, f"ollama unreachable: {exc}"


def build_generator() -> BuiltGenerator:
    """Build the generation pipeline or a disabled stub if Ollama is off."""
    if not SETTINGS.enable_generation:
        logger.info("Generation disabled via EXPERIMENTS_ENABLE_GENERATION=0")
        return BuiltGenerator(
            strategy=None, available=False,
            reason="EXPERIMENTS_ENABLE_GENERATION=0",
            model_name=SETTINGS.ollama_model,
        )

    ok, reason = _probe_ollama(SETTINGS.ollama_base_url, SETTINGS.ollama_timeout_s)
    if not ok:
        logger.warning(
            "Ollama probe failed (%s); generation suites will record skips",
            reason,
        )
        return BuiltGenerator(
            strategy=None, available=False,
            reason=reason, model_name=SETTINGS.ollama_model,
        )

    cfg = GenerationConfig(
        model_name=SETTINGS.ollama_model,
        base_url=SETTINGS.ollama_base_url,
        temperature=0.0,
        max_tokens=256,
        seed=SETTINGS.seed,
        timeout=SETTINGS.ollama_timeout_s,
        max_retries=1,
        retry_delay=1.0,
        max_context_chars=4000,
    )
    client = OllamaClient(cfg)
    builder = PromptBuilder(STRICT_RAG_TEMPLATE)
    preparer = ContextPreparer(max_context_chars=cfg.max_context_chars)
    strategy = SimpleRAGStrategy(client, builder, preparer)

    logger.info("Generation enabled: model=%s url=%s", cfg.model_name, cfg.base_url)
    return BuiltGenerator(
        strategy=strategy, available=True, reason="",
        model_name=cfg.model_name,
    )
