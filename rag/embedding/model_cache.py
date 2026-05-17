import threading
from typing import Any, Callable, Dict, List


class ModelCache:
    """Thread-safe LRU cache for loaded model objects.

    Key format: "{model_name}::{device}::{dtype}". Holds the lock during loading
    so concurrent callers don't trigger duplicate GPU allocations.
    """

    def __init__(self, max_size: int = 8) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()

    @staticmethod
    def make_key(model_name: str, device: str, dtype: str = "float32") -> str:
        return f"{model_name}::{device}::{dtype}"

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        """Return cached model or call loader() and cache the result."""
        with self._lock:
            if key in self._cache:
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]

            model = loader()

            if len(self._cache) >= self._max_size:
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

            self._cache[key] = model
            self._access_order.append(key)
            return model

    def evict(self, key: str) -> None:
        """Remove an entry to free GPU/RAM."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._access_order.remove(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def cached_keys(self) -> List[str]:
        """Return current keys in LRU order (least recently used first)."""
        with self._lock:
            return list(self._access_order)


_default_cache = ModelCache(max_size=8)


def get_default_cache() -> ModelCache:
    """Return the process-level default cache."""
    return _default_cache
