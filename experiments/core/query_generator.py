"""Synthetic query generation with provenance-anchored gold relevance.

Because no human-labelled QA dataset exists for the corpora, the framework
generates evaluation queries directly from chunks. The chunk a query was
derived from is treated as the gold-relevant chunk. Where a query is built
from several chunks (e.g. multi-hop), all source chunks count as gold.

Five stratified query types are produced:

* keyword              short noun-phrase / capitalised-term queries
* semantic_paraphrase  reworded full sentences from a chunk
* ambiguous            short truncated heads with little context
* noisy                queries with injected typos / token drops
* multi_hop            queries joining facets from two distinct chunks

The generator is deterministic given a seed so reruns reproduce the eval set.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
from typing import Dict, List, Optional

from experiments.core.types import QUERY_TYPES, SyntheticQuery

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_WORD = re.compile(r"\b[\w/-]+\b")
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "at", "by", "from", "into", "than",
    "then", "we", "you", "they", "he", "she", "his", "her", "their", "our",
    "but", "not", "no", "do", "does", "did", "has", "have", "had", "will",
    "would", "should", "could", "can", "may", "might", "such", "also", "if",
    "so", "which", "who", "whom", "what", "when", "where", "why", "how",
    "any", "all", "some", "more", "most", "very", "much", "many",
})


def _sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if len(p.split()) >= 4]


def _content_words(text: str, min_len: int = 4) -> List[str]:
    return [
        w for w in _WORD.findall(text.lower())
        if len(w) >= min_len and w not in _STOPWORDS
    ]


def _capitalised_phrases(text: str) -> List[str]:
    """Extract sequences of capitalised tokens (likely named entities)."""
    pattern = re.compile(r"\b([A-Z][\w/-]+(?:\s+[A-Z][\w/-]+){0,3})\b")
    return [m.group(1).strip() for m in pattern.finditer(text)]


def _shorten(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _query_id(query_type: str, salt: str) -> str:
    digest = hashlib.sha1(f"{query_type}:{salt}".encode("utf-8")).hexdigest()
    return f"q_{query_type[:4]}_{digest[:10]}"


def _build_keyword_query(chunk: Dict, rng: random.Random) -> Optional[SyntheticQuery]:
    """Build a short keyword-style query from named entities / content words."""
    text = chunk["text"]
    caps = _capitalised_phrases(text)
    if caps:
        rng.shuffle(caps)
        terms = caps[: rng.randint(1, 2)]
        words: List[str] = []
        for term in terms:
            words.extend(term.split())
        query = " ".join(words)
    else:
        content = _content_words(text)
        if len(content) < 3:
            return None
        rng.shuffle(content)
        query = " ".join(content[: rng.randint(2, 4)])

    query = query.strip()
    if len(query) < 3:
        return None

    return SyntheticQuery(
        query_id=_query_id("keyword", chunk["id"] + query),
        text=query,
        query_type="keyword",
        corpus="",
        relevant_chunk_ids=[chunk["id"]],
        relevant_document_ids=[chunk["document_id"]],
        source_chunk_excerpt=_shorten(text, 220),
        expected_answer=_shorten(text, 220),
    )


def _build_semantic_paraphrase(
    chunk: Dict, rng: random.Random
) -> Optional[SyntheticQuery]:
    """Rephrase a sentence from the chunk as a natural-language question."""
    sentences = _sentences(chunk["text"])
    if not sentences:
        return None
    sentence = rng.choice(sentences)
    text = sentence.strip().rstrip(".")
    # Light paraphrasing rules: convert declaratives into questions.
    prefixes = [
        "Explain what is meant by:",
        "In your own words, describe:",
        "What is the role of",
        "Summarise the following:",
        "Why does the text state that",
    ]
    prefix = rng.choice(prefixes)
    query = f"{prefix} {text}?"

    return SyntheticQuery(
        query_id=_query_id("semantic_paraphrase", chunk["id"] + sentence),
        text=query,
        query_type="semantic_paraphrase",
        corpus="",
        relevant_chunk_ids=[chunk["id"]],
        relevant_document_ids=[chunk["document_id"]],
        source_chunk_excerpt=_shorten(chunk["text"], 220),
        expected_answer=_shorten(sentence, 220),
    )


def _build_ambiguous_query(
    chunk: Dict, rng: random.Random
) -> Optional[SyntheticQuery]:
    """Take a short truncated head of a sentence — low-context, multiple matches."""
    sentences = _sentences(chunk["text"])
    if not sentences:
        return None
    sentence = rng.choice(sentences)
    words = sentence.split()
    if len(words) < 5:
        return None
    head = " ".join(words[: rng.randint(2, 4)])
    if len(head) < 5:
        return None

    return SyntheticQuery(
        query_id=_query_id("ambiguous", chunk["id"] + head),
        text=head,
        query_type="ambiguous",
        corpus="",
        relevant_chunk_ids=[chunk["id"]],
        relevant_document_ids=[chunk["document_id"]],
        source_chunk_excerpt=_shorten(chunk["text"], 220),
        expected_answer=_shorten(sentence, 220),
    )


def _inject_noise(text: str, rng: random.Random) -> str:
    """Inject mild typos and token drops; preserves overall semantics."""
    tokens = text.split()
    if not tokens:
        return text
    n_ops = max(1, len(tokens) // 6)
    for _ in range(n_ops):
        i = rng.randrange(len(tokens))
        word = tokens[i]
        if len(word) < 4:
            continue
        op = rng.choice(("swap", "drop_char", "drop_token", "duplicate"))
        if op == "swap" and len(word) > 3:
            j = rng.randrange(len(word) - 1)
            chars = list(word)
            chars[j], chars[j + 1] = chars[j + 1], chars[j]
            tokens[i] = "".join(chars)
        elif op == "drop_char":
            j = rng.randrange(len(word))
            tokens[i] = word[:j] + word[j + 1:]
        elif op == "drop_token":
            tokens[i] = ""
        elif op == "duplicate":
            tokens.insert(i, word)
    return " ".join(t for t in tokens if t)


def _build_noisy_query(
    chunk: Dict, rng: random.Random
) -> Optional[SyntheticQuery]:
    base = _build_semantic_paraphrase(chunk, rng)
    if base is None:
        return None
    noisy = _inject_noise(base.text, rng)
    if not noisy.strip():
        return None
    return SyntheticQuery(
        query_id=_query_id("noisy", base.query_id + noisy),
        text=noisy,
        query_type="noisy",
        corpus="",
        relevant_chunk_ids=base.relevant_chunk_ids,
        relevant_document_ids=base.relevant_document_ids,
        source_chunk_excerpt=base.source_chunk_excerpt,
        expected_answer=base.expected_answer,
    )


def _build_multihop_query(
    chunk_a: Dict, chunk_b: Dict, rng: random.Random
) -> Optional[SyntheticQuery]:
    """Combine entities/words from two different chunks."""
    a_caps = _capitalised_phrases(chunk_a["text"]) or _content_words(chunk_a["text"])
    b_caps = _capitalised_phrases(chunk_b["text"]) or _content_words(chunk_b["text"])
    if not a_caps or not b_caps:
        return None
    a = rng.choice(a_caps)
    b = rng.choice(b_caps)
    templates = (
        "How does {a} relate to {b}?",
        "Compare {a} and {b}.",
        "What is the connection between {a} and {b}?",
    )
    query = rng.choice(templates).format(a=a, b=b)

    return SyntheticQuery(
        query_id=_query_id("multi_hop", chunk_a["id"] + chunk_b["id"]),
        text=query,
        query_type="multi_hop",
        corpus="",
        relevant_chunk_ids=[chunk_a["id"], chunk_b["id"]],
        relevant_document_ids=list({chunk_a["document_id"], chunk_b["document_id"]}),
        source_chunk_excerpt=_shorten(chunk_a["text"], 110) + " || " + _shorten(chunk_b["text"], 110),
        expected_answer="",
    )


def generate_queries(
    chunks: List[Dict],
    corpus_name: str,
    n_total: int,
    seed: int,
) -> List[SyntheticQuery]:
    """Generate ``n_total`` queries balanced across the five stratified types.

    Falls back gracefully if a chunk cannot yield a particular type.
    """
    if not chunks:
        raise ValueError("Cannot generate queries from an empty chunk list.")

    rng = random.Random(seed)
    per_type = max(1, n_total // len(QUERY_TYPES))
    queries: List[SyntheticQuery] = []
    seen_ids: set = set()

    type_builders = {
        "keyword": _build_keyword_query,
        "semantic_paraphrase": _build_semantic_paraphrase,
        "ambiguous": _build_ambiguous_query,
        "noisy": _build_noisy_query,
    }

    for qtype, builder in type_builders.items():
        attempts = 0
        produced = 0
        order = list(range(len(chunks)))
        rng.shuffle(order)
        for idx in order:
            if produced >= per_type or attempts >= per_type * 8:
                break
            attempts += 1
            chunk = chunks[idx]
            q = builder(chunk, rng)
            if q is None:
                continue
            if q.query_id in seen_ids:
                continue
            seen_ids.add(q.query_id)
            queries.append(
                SyntheticQuery(
                    query_id=q.query_id,
                    text=q.text,
                    query_type=q.query_type,
                    corpus=corpus_name,
                    relevant_chunk_ids=q.relevant_chunk_ids,
                    relevant_document_ids=q.relevant_document_ids,
                    source_chunk_excerpt=q.source_chunk_excerpt,
                    expected_answer=q.expected_answer,
                    metadata={"seed": seed},
                )
            )
            produced += 1

    # Multi-hop requires two distinct documents.
    by_doc: Dict[str, List[Dict]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["document_id"], []).append(chunk)
    doc_ids = list(by_doc.keys())
    multihop_produced = 0
    multihop_attempts = 0
    while multihop_produced < per_type and multihop_attempts < per_type * 10:
        multihop_attempts += 1
        if len(doc_ids) < 2:
            break
        doc_a, doc_b = rng.sample(doc_ids, 2)
        chunk_a = rng.choice(by_doc[doc_a])
        chunk_b = rng.choice(by_doc[doc_b])
        q = _build_multihop_query(chunk_a, chunk_b, rng)
        if q is None or q.query_id in seen_ids:
            continue
        seen_ids.add(q.query_id)
        queries.append(
            SyntheticQuery(
                query_id=q.query_id,
                text=q.text,
                query_type="multi_hop",
                corpus=corpus_name,
                relevant_chunk_ids=q.relevant_chunk_ids,
                relevant_document_ids=q.relevant_document_ids,
                source_chunk_excerpt=q.source_chunk_excerpt,
                expected_answer=q.expected_answer,
                metadata={"seed": seed},
            )
        )
        multihop_produced += 1

    logger.info(
        "Generated %d queries for %s (target=%d, types=%s)",
        len(queries),
        corpus_name,
        n_total,
        {
            qt: sum(1 for q in queries if q.query_type == qt)
            for qt in QUERY_TYPES
        },
    )
    return queries
