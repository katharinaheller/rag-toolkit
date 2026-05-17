import re
from typing import List

from rag.indexing.config import SparseIndexConfig


class Tokenizer:
    """Deterministic text tokenizer for indexing and retrieval.

    Must be used identically at index build time and query time. Using different
    configurations for each produces undefined retrieval behavior.
    """

    def __init__(self, config: SparseIndexConfig) -> None:
        self._config = config

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase terms.

        'simple' uses \\b\\w+\\b (handles punctuation); 'whitespace' splits on whitespace.
        """
        if self._config.tokenizer == "simple":
            return re.findall(r"\b\w+\b", text.lower())
        if self._config.tokenizer == "whitespace":
            return text.lower().split()
        raise ValueError(f"Unknown tokenizer: '{self._config.tokenizer}'.")
