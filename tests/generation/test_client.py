"""Tests for OllamaClient covering retries, error translation, and payload shape."""

from __future__ import annotations

import requests

import pytest

from rag.generation.client import OllamaClient
from rag.generation.config import GenerationConfig
from rag.generation.exceptions import (
    LLMConnectionError,
    LLMResponseError,
    LLMTimeoutError,
)
from tests.mocks.ollama import (
    ConnectionErrorSequence,
    FakeResponse,
    TimeoutSequence,
    error_response,
    http_error_response,
    install_requests_mock,
    malformed_json_response,
    missing_response_field,
    ok_response,
)


@pytest.fixture
def fast_config() -> GenerationConfig:
    """Use zero retry delay for fast test execution."""
    return GenerationConfig(model_name="mistral", max_retries=2, retry_delay=0.0, timeout=1.0)


@pytest.fixture
def no_retry_config() -> GenerationConfig:
    return GenerationConfig(model_name="mistral", max_retries=0, retry_delay=0.0, timeout=1.0)


class TestPayload:
    def test_payload_contains_required_fields(self, fast_config, monkeypatch) -> None:
        calls = install_requests_mock(monkeypatch, lambda *a, **kw: ok_response("answer"))
        OllamaClient(fast_config).generate("hello")
        sent = calls[0]["json"]
        assert sent["model"] == "mistral"
        assert sent["prompt"] == "hello"
        assert sent["stream"] is False
        assert "options" in sent
        assert sent["options"]["temperature"] == 0.0

    def test_request_uses_configured_url_and_timeout(self, fast_config, monkeypatch) -> None:
        calls = install_requests_mock(monkeypatch, lambda *a, **kw: ok_response())
        OllamaClient(fast_config).generate("hello")
        assert calls[0]["url"] == fast_config.url
        assert calls[0]["timeout"] == fast_config.timeout


class TestSuccessPath:
    def test_returns_decoded_response(self, fast_config, monkeypatch) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: ok_response("Hi there"))
        body = OllamaClient(fast_config).generate("hi")
        assert body["response"] == "Hi there"


class TestRetriesOnTransient:
    def test_retries_then_succeeds_on_timeout(self, fast_config, monkeypatch) -> None:
        seq = TimeoutSequence(n_timeouts=2, then=ok_response("done"))
        install_requests_mock(monkeypatch, seq)
        body = OllamaClient(fast_config).generate("p")
        assert body["response"] == "done"
        assert seq.calls == 3  # initial + 2 retries used

    def test_retries_then_succeeds_on_connection_error(self, fast_config, monkeypatch) -> None:
        seq = ConnectionErrorSequence(n_errors=1, then=ok_response("ok"))
        install_requests_mock(monkeypatch, seq)
        body = OllamaClient(fast_config).generate("p")
        assert body["response"] == "ok"

    def test_raises_timeout_after_exhausting_retries(self, fast_config, monkeypatch) -> None:
        seq = TimeoutSequence(n_timeouts=10)
        install_requests_mock(monkeypatch, seq)
        with pytest.raises(LLMTimeoutError):
            OllamaClient(fast_config).generate("p")

    def test_raises_connection_error_after_exhausting_retries(
        self, fast_config, monkeypatch
    ) -> None:
        seq = ConnectionErrorSequence(n_errors=10)
        install_requests_mock(monkeypatch, seq)
        with pytest.raises(LLMConnectionError):
            OllamaClient(fast_config).generate("p")


class TestNoRetryOnResponseErrors:
    def test_http_error_does_not_retry(self, no_retry_config, monkeypatch) -> None:
        calls = install_requests_mock(monkeypatch, lambda *a, **kw: http_error_response(500))
        with pytest.raises(LLMResponseError, match="HTTP 500"):
            OllamaClient(no_retry_config).generate("p")
        assert len(calls) == 1

    def test_ollama_error_payload_raises_response_error(
        self, no_retry_config, monkeypatch
    ) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: error_response("model missing"))
        with pytest.raises(LLMResponseError, match="model missing"):
            OllamaClient(no_retry_config).generate("p")

    def test_missing_response_field_raises(self, no_retry_config, monkeypatch) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: missing_response_field())
        with pytest.raises(LLMResponseError, match="missing 'response'"):
            OllamaClient(no_retry_config).generate("p")

    def test_malformed_json_raises(self, no_retry_config, monkeypatch) -> None:
        install_requests_mock(monkeypatch, lambda *a, **kw: malformed_json_response())
        with pytest.raises(LLMResponseError, match="not valid JSON"):
            OllamaClient(no_retry_config).generate("p")


class TestNoRetriesWhenDisabled:
    def test_max_retries_zero_makes_single_attempt(self, no_retry_config, monkeypatch) -> None:
        seq = TimeoutSequence(n_timeouts=10)
        install_requests_mock(monkeypatch, seq)
        with pytest.raises(LLMTimeoutError):
            OllamaClient(no_retry_config).generate("p")
        assert seq.calls == 1
