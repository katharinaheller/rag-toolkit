"""Lightweight stand-ins for `requests` and Ollama responses.

Mocking happens at the `requests.post` boundary so that all
exception-translation logic inside `OllamaClient` is exercised by tests
exactly as it would be in production.
"""

from __future__ import annotations

import json
from typing import Any, Callable, List, Optional

import requests


class FakeResponse:
    """Minimal `requests.Response`-compatible object."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        body: Optional[dict] = None,
        text: Optional[str] = None,
    ) -> None:
        self.status_code = status_code
        if text is not None:
            self.text = text
            self._body = None
        else:
            self._body = body if body is not None else {"response": "ok"}
            self.text = json.dumps(self._body)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        if self._body is None:
            raise ValueError("not json")
        return self._body


def install_requests_mock(monkeypatch, side_effect: Callable[..., Any]) -> List[dict]:
    """Replace `requests.post` with side_effect; return a list capturing calls.

    Each captured entry is the kwargs dict passed to requests.post.
    """
    calls: List[dict] = []

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        result = side_effect(url, **kwargs)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


def ok_response(answer: str = "Generated answer.") -> FakeResponse:
    """Return a successful Ollama-shaped response."""
    return FakeResponse(body={"response": answer, "done": True})


def error_response(message: str = "model not found") -> FakeResponse:
    """Return a 200 OK with an embedded Ollama error payload."""
    return FakeResponse(body={"error": message})


def http_error_response(status: int = 500) -> FakeResponse:
    """Return a non-OK HTTP response."""
    return FakeResponse(status_code=status, text="server error")


def malformed_json_response() -> FakeResponse:
    """Return a response whose body is not parseable JSON."""
    return FakeResponse(text="not json at all")


def missing_response_field() -> FakeResponse:
    """Return a JSON body that lacks the required 'response' key."""
    return FakeResponse(body={"done": True})


class TimeoutSequence:
    """Returns timeout exceptions for n calls, then OK responses."""

    def __init__(self, n_timeouts: int = 2, then: Optional[FakeResponse] = None) -> None:
        self.n_timeouts = n_timeouts
        self.then = then or ok_response()
        self.calls = 0

    def __call__(self, url, **kwargs):
        self.calls += 1
        if self.calls <= self.n_timeouts:
            return requests.exceptions.Timeout("timed out")
        return self.then


class ConnectionErrorSequence:
    """Returns ConnectionError for n calls, then OK responses."""

    def __init__(self, n_errors: int = 2, then: Optional[FakeResponse] = None) -> None:
        self.n_errors = n_errors
        self.then = then or ok_response()
        self.calls = 0

    def __call__(self, url, **kwargs):
        self.calls += 1
        if self.calls <= self.n_errors:
            return requests.exceptions.ConnectionError("refused")
        return self.then
