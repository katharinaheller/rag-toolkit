from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class BaseStore(ABC):

    @abstractmethod
    def write_many(self, items: Iterable[Dict[str, Any]]) -> None:
        pass
