"""Behavioural tests for SimpleRAGStrategy and RefineRAGStrategy."""

from __future__ import annotations

import pytest

from rag.generation.client import OllamaClient
from rag.generation.config import GenerationConfig
from rag.generation.context import ContextPreparer
from rag.generation.prompt_builder import STRICT_RAG_TEMPLATE, PromptBuilder
from rag.generation.strategies import RefineRAGStrategy, SimpleRAGStrategy
from tests.mocks.ollama import (
    error_response,
    http_error_response,
    install_requests_mock,
    ok_response,
)


@pytest.fixture
def cfg() -> GenerationConfig:
    return GenerationConfig(
        model_name="mistral", max_retries=0, retry_delay=0.0, timeout=1.0,
        max_context_chars=200,
    )


@pytest.fixture
def client(cfg: GenerationConfig) -> OllamaClient:
    return OllamaClient(cfg)


@pytest.fixture
def preparer(cfg: GenerationConfig) -> ContextPreparer:
    return ContextPreparer(cfg.max_context_chars)


@pytest.fixture
def builder() -> PromptBuilder:
    return PromptBuilder(STRICT_RAG_TEMPLATE)


class TestSimpleRAGStrategy:
    def test_success_path_returns_answer_and_metadata(
        self, client, preparer, builder, monkeypatch
    ) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: ok_response("Final answer"))
        strategy = SimpleRAGStrategy(client, builder, preparer)
        result = strategy.generate("What is X?", ["chunk one"])
        assert result.answer == "Final answer"
        assert result.error is None
        assert result.success is True
        assert result.model == "mistral"
        assert result.template_name == STRICT_RAG_TEMPLATE.name
        assert result.strategy == "SimpleRAGStrategy"
        assert result.prompt_chars > 0
        assert result.context_chars > 0
        assert result.latency_ms >= 0.0

    def test_failure_returns_error_with_empty_answer(
        self, client, preparer, builder, monkeypatch
    ) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: error_response("boom"))
        strategy = SimpleRAGStrategy(client, builder, preparer)
        result = strategy.generate("q", ["c"])
        assert result.answer == ""
        assert result.error is not None
        assert result.success is False

    def test_unexpected_exception_captured(
        self, client, preparer, builder, monkeypatch
    ) -> None:
        def boom(url, **kw):
            raise RuntimeError("totally unexpected")
        install_requests_mock(monkeypatch, boom)
        strategy = SimpleRAGStrategy(client, builder, preparer)
        result = strategy.generate("q", ["c"])
        assert result.error is not None
        assert "Unexpected error" in result.error or "totally unexpected" in result.error

    def test_empty_context_still_produces_prompt(
        self, client, preparer, builder, monkeypatch
    ) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: ok_response("no context answer"))
        strategy = SimpleRAGStrategy(client, builder, preparer)
        result = strategy.generate("q", [])
        assert result.context_chars == 0
        assert result.success

    def test_inference_parameters_recorded(
        self, client, preparer, builder, monkeypatch
    ) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: ok_response())
        strategy = SimpleRAGStrategy(client, builder, preparer)
        result = strategy.generate("q", ["c"])
        assert "temperature" in result.inference_parameters
        assert "num_predict" in result.inference_parameters


class TestRefineRAGStrategy:
    def test_calls_llm_once_per_chunk(
        self, client, preparer, monkeypatch
    ) -> None:
        calls = install_requests_mock(monkeypatch, lambda *a, **kw: ok_response("refined"))
        strategy = RefineRAGStrategy(client, preparer)
        result = strategy.generate("q", ["a", "b", "c"])
        assert result.success
        assert len(calls) == 3

    def test_empty_context_returns_error_result(
        self, client, preparer, monkeypatch
    ) -> None:
        calls = install_requests_mock(monkeypatch, lambda *a, **kw: ok_response("ignored"))
        strategy = RefineRAGStrategy(client, preparer)
        result = strategy.generate("q", [])
        assert result.error is not None
        assert "No valid context" in result.error
        assert len(calls) == 0

    def test_failure_during_initial_pass_short_circuits(
        self, client, preparer, monkeypatch
    ) -> None:
        calls = install_requests_mock(monkeypatch, lambda *a, **kw: http_error_response(500))
        strategy = RefineRAGStrategy(client, preparer)
        result = strategy.generate("q", ["a", "b"])
        assert result.error is not None
        assert len(calls) == 1

    def test_failure_during_refine_returns_error(
        self, client, preparer, monkeypatch
    ) -> None:
        n = {"i": 0}
        def fake(url, **kw):
            n["i"] += 1
            return ok_response("first") if n["i"] == 1 else http_error_response(500)
        install_requests_mock(monkeypatch, fake)
        strategy = RefineRAGStrategy(client, preparer)
        result = strategy.generate("q", ["a", "b"])
        assert result.error is not None
        assert n["i"] == 2

    def test_final_answer_is_last_refinement(
        self, client, preparer, monkeypatch
    ) -> None:
        n = {"i": 0}
        def fake(url, **kw):
            n["i"] += 1
            return ok_response(f"answer-{n['i']}")
        install_requests_mock(monkeypatch, fake)
        strategy = RefineRAGStrategy(client, preparer)
        result = strategy.generate("q", ["a", "b", "c"])
        assert result.success
        assert result.answer == "answer-3"
