import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Set

from rag.indexing.types import InvertedIndexStats, TokenizedEntry


class InvertedIndex:
    """Pure lexical inverted index. No scoring, no text handling.

    All methods are prefixed with _ — external code uses InvertedIndexView.

    Storage:
        _dictionary:    token -> {doc_id: tf}
        _doc_lengths:   doc_id -> token count
        _doc_frequency: token -> doc count
        _vocabulary:    token -> integer id
    """

    def __init__(self) -> None:
        self._dictionary: Dict[str, Dict[str, int]] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._doc_frequency: Dict[str, int] = {}
        self._vocabulary: Dict[str, int] = {}

    def _reset(self) -> None:
        self._dictionary = {}
        self._doc_lengths = {}
        self._doc_frequency = {}
        self._vocabulary = {}

    def _build(self, entries: Iterable[TokenizedEntry]) -> None:
        """Ingest pre-tokenized entries. Additive: call _reset() before a full rebuild."""
        df_acc: DefaultDict[str, int] = defaultdict(int)

        for entry in entries:
            doc_id = entry["id"]
            tokens = entry["tokens"]

            if not isinstance(doc_id, str) or not doc_id:
                raise ValueError("TokenizedEntry.id must be a non-empty string.")
            if doc_id in self._doc_lengths:
                raise ValueError(f"Duplicate tokenized document id: '{doc_id}'.")
            if not isinstance(tokens, list):
                raise TypeError("TokenizedEntry.tokens must be List[str].")
            if any(not isinstance(t, str) for t in tokens):
                raise TypeError("TokenizedEntry.tokens must contain only strings.")

            self._doc_lengths[doc_id] = len(tokens)
            tf_counter = Counter(tokens)
            seen: Set[str] = set()

            for token, tf in tf_counter.items():
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)
                if token not in self._dictionary:
                    self._dictionary[token] = {}
                self._dictionary[token][doc_id] = tf
                if token not in seen:
                    df_acc[token] += 1
                    seen.add(token)

        for token, count in df_acc.items():
            self._doc_frequency[token] = self._doc_frequency.get(token, 0) + count

    def _get_tf(self, token: str, doc_id: str) -> int:
        """Term frequency in O(1). Returns 0 if absent."""
        return self._dictionary.get(token, {}).get(doc_id, 0)

    def _get_matching_doc_ids(self, tokens: List[str]) -> Set[str]:
        """All doc IDs containing at least one of the given tokens."""
        ids: Set[str] = set()
        for token in tokens:
            ids.update(self._dictionary.get(token, {}).keys())
        return ids

    def _get_stats(self) -> InvertedIndexStats:
        """Corpus statistics for BM25/TF-IDF scoring."""
        n = len(self._doc_lengths)
        avg = sum(self._doc_lengths.values()) / n if n > 0 else 0.0
        return {
            "doc_count": n,
            "avg_doc_length": avg,
            "doc_lengths": dict(self._doc_lengths),
            "document_frequency": dict(self._doc_frequency),
        }

    def _get_vocabulary(self) -> Dict[str, int]:
        return dict(self._vocabulary)

    def _persist(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        _write_json(index_dir / "dictionary.json", self._dictionary)
        _write_json(
            index_dir / "stats.json",
            {"doc_lengths": self._doc_lengths, "document_frequency": self._doc_frequency},
        )
        _write_json(index_dir / "vocabulary.json", self._vocabulary)

    def _load(self, index_dir: Path) -> None:
        d = index_dir / "dictionary.json"
        s = index_dir / "stats.json"
        v = index_dir / "vocabulary.json"

        self._reset()

        if d.exists():
            self._dictionary = _read_json(d)
        if s.exists():
            data = _read_json(s)
            self._doc_lengths = data.get("doc_lengths", {})
            self._doc_frequency = data.get("document_frequency", {})
        if v.exists():
            self._vocabulary = _read_json(v)


def _write_json(path: Path, obj: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
