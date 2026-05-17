import math
import re
from typing import List, Set

from rag.retrieval.config import RerankerConfig
from rag.retrieval.types import RetrievalResult


class BaselineReranker:
    """Simple lexical + length baseline reranker.

    Signals:
        lexical_overlap = |query_tokens ∩ doc_tokens| / |query_tokens|
        length_score    = exp(-((len(text) - target)²) / (2 * target²))

    Score update (when enabled):
        rerank_delta = lexical_weight * lexical_overlap + length_weight * length_score
        result.score = result.retrieval_score + rerank_delta

    When disabled: returns input list unchanged.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not self._config.enabled:
            return results
        if not results:
            return results

        query_tokens: Set[str] = set(_simple_tokenize(query))

        reranked: List[RetrievalResult] = []
        for result in results:
            retrieval_score = result["retrieval_score"]
            text = result["text"]

            lex_overlap = self._lexical_overlap(query_tokens, text)
            len_score = self._length_score(text)
            rerank_delta = (
                self._config.lexical_weight * lex_overlap
                + self._config.length_weight * len_score
            )
            final_score = retrieval_score + rerank_delta

            entry: RetrievalResult = {
                "id": result["id"],
                "chunk_id": result["chunk_id"],
                "document_id": result["document_id"],
                "score": final_score,
                "retrieval_score": retrieval_score,
                "rerank_score": rerank_delta,
                "text": text,
                "metadata": result["metadata"],
            }
            if "dense_score" in result:
                entry["dense_score"] = result["dense_score"]  # type: ignore[typeddict-unknown-key]
            if "sparse_score" in result:
                entry["sparse_score"] = result["sparse_score"]  # type: ignore[typeddict-unknown-key]

            reranked.append(entry)

        reranked.sort(key=lambda r: (-r["score"], r["id"]))
        return reranked

    def _lexical_overlap(self, query_tokens: Set[str], text: str) -> float:
        if not query_tokens:
            return 0.0
        doc_tokens: Set[str] = set(_simple_tokenize(text))
        return len(query_tokens & doc_tokens) / len(query_tokens)

    def _length_score(self, text: str) -> float:
        target = self._config.target_length
        exponent = -((len(text) - target) ** 2) / (2.0 * (target ** 2))
        return math.exp(exponent)


def _simple_tokenize(text: str) -> List[str]:
    r"""Extract lowercase word tokens. Same regex as Tokenizer(tokenizer='simple')."""
    return re.findall(r"\b\w+\b", text.lower())
