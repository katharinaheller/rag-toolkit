"""Concurrency and LRU correctness tests for ModelCache."""

from __future__ import annotations

import threading
import time

import pytest

from rag.embedding.model_cache import ModelCache, get_default_cache


def test_make_key_includes_all_components() -> None:
    key = ModelCache.make_key("BAAI/bge-m3", "cuda:0", "fp16")
    assert key == "BAAI/bge-m3::cuda:0::fp16"


def test_make_key_defaults_dtype_to_float32() -> None:
    assert ModelCache.make_key("m", "cpu") == "m::cpu::float32"


def test_rejects_non_positive_max_size() -> None:
    with pytest.raises(ValueError, match="max_size must be positive"):
        ModelCache(max_size=0)


def test_get_or_load_calls_loader_only_once() -> None:
    cache = ModelCache(max_size=4)
    calls = []

    def loader():
        calls.append(1)
        return object()

    a = cache.get_or_load("k", loader)
    b = cache.get_or_load("k", loader)
    assert a is b
    assert len(calls) == 1


def test_lru_eviction_removes_oldest_entry() -> None:
    cache = ModelCache(max_size=2)
    cache.get_or_load("a", lambda: "A")
    cache.get_or_load("b", lambda: "B")
    cache.get_or_load("c", lambda: "C")  # evicts "a"
    assert "a" not in cache.cached_keys()
    assert set(cache.cached_keys()) == {"b", "c"}


def test_access_moves_key_to_most_recent() -> None:
    cache = ModelCache(max_size=2)
    cache.get_or_load("a", lambda: 1)
    cache.get_or_load("b", lambda: 2)
    cache.get_or_load("a", lambda: 1)  # refresh "a"
    cache.get_or_load("c", lambda: 3)  # evict the LRU which is "b"
    assert "b" not in cache.cached_keys()
    assert set(cache.cached_keys()) == {"a", "c"}


def test_evict_removes_specific_key() -> None:
    cache = ModelCache(max_size=2)
    cache.get_or_load("x", lambda: 1)
    cache.evict("x")
    assert "x" not in cache.cached_keys()


def test_evict_unknown_key_is_silent() -> None:
    cache = ModelCache(max_size=2)
    cache.evict("missing")  # must not raise


def test_clear_empties_cache() -> None:
    cache = ModelCache(max_size=2)
    cache.get_or_load("x", lambda: 1)
    cache.get_or_load("y", lambda: 2)
    cache.clear()
    assert cache.cached_keys() == []


def test_get_or_load_is_thread_safe_for_concurrent_loads() -> None:
    cache = ModelCache(max_size=4)
    barrier = threading.Barrier(8)
    load_counter = []

    def loader():
        load_counter.append(1)
        time.sleep(0.01)
        return "model"

    results = []

    def worker():
        barrier.wait()
        results.append(cache.get_or_load("shared", loader))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(load_counter) == 1, "loader was called more than once under contention"
    assert all(r == "model" for r in results)


def test_default_cache_singleton_is_stable() -> None:
    assert get_default_cache() is get_default_cache()
