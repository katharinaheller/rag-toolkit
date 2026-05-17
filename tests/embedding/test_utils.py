"""Deterministic embedding ID derivation tests."""

from __future__ import annotations

import hashlib

import pytest

from rag.embedding.utils import embedding_id


def test_id_is_sha256_of_chunk_model_version() -> None:
    expected = hashlib.sha256("c1|m1|v1".encode("utf-8")).hexdigest()
    assert embedding_id("c1", "m1", "v1") == expected


def test_id_changes_with_chunk_id() -> None:
    a = embedding_id("c1", "m", "v")
    b = embedding_id("c2", "m", "v")
    assert a != b


def test_id_changes_with_model_version() -> None:
    a = embedding_id("c1", "m", "v1")
    b = embedding_id("c1", "m", "v2")
    assert a != b


def test_id_changes_with_model_name() -> None:
    a = embedding_id("c", "m1", "v")
    b = embedding_id("c", "m2", "v")
    assert a != b


def test_no_version_uses_unversioned_tag() -> None:
    expected = hashlib.sha256("c|m|unversioned".encode("utf-8")).hexdigest()
    assert embedding_id("c", "m") == expected
    assert embedding_id("c", "m", None) == expected


def test_id_is_deterministic_across_calls() -> None:
    a = embedding_id("c", "m", "v")
    b = embedding_id("c", "m", "v")
    assert a == b


def test_id_handles_unicode_safely() -> None:
    out = embedding_id("c_ünicodé", "m_中文", "v")
    assert isinstance(out, str)
    assert len(out) == 64
