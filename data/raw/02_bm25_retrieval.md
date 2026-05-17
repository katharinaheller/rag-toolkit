# BM25: Best Match 25

## Overview

BM25, an abbreviation for Best Match 25, is a probabilistic bag-of-words ranking function used extensively in information retrieval. It was developed within the Okapi project at City University of London during the 1990s by Stephen Robertson, Karen Sparck Jones, and colleagues. The "25" in the name refers to the 25th iteration of the retrieval model, developed through extensive experimentation.

BM25 is the de-facto baseline for lexical (sparse) retrieval in modern information retrieval systems and remains competitive with many neural approaches, particularly when query terms overlap strongly with document vocabulary.

## Mathematical Formulation

Given a query Q containing terms q1, q2, ..., qn and a document D, the BM25 score is:

```
score(D, Q) = Σ IDF(qi) × TF_sat(qi, D)
```

where the Inverse Document Frequency (IDF) component is:

```
IDF(qi) = log( (N - df(qi) + 0.5) / (df(qi) + 0.5) + 1 )
```

and the saturated Term Frequency (TF) component with length normalisation is:

```
TF_sat(qi, D) = (tf(qi, D) × (k1 + 1)) / (tf(qi, D) + k1 × (1 - b + b × (|D| / avgdl)))
```

### Parameters

- **N**: Total number of documents in the corpus.
- **df(qi)**: Number of documents containing term qi.
- **tf(qi, D)**: Raw term frequency of qi in document D.
- **|D|**: Length of document D in tokens.
- **avgdl**: Average document length across the corpus.
- **k1**: Term frequency saturation parameter. Typical value: 1.2–1.5. Higher values allow term frequency to grow without bound; lower values saturate quickly.
- **b**: Length normalisation parameter in [0, 1]. b=0 disables length normalisation; b=1 applies full normalisation. Typical value: 0.75.

## Comparison with TF-IDF

BM25 generalises and improves over classical TF-IDF scoring in two key ways:

1. **TF saturation**: BM25 saturates the term frequency contribution via the k1 parameter, preventing very high-frequency terms from dominating the score. TF-IDF applies no saturation.
2. **Length normalisation**: BM25 normalises TF scores by document length relative to the corpus average, controlled by the b parameter. This penalises long documents that accumulate high raw TF values by chance.

## Implementation Notes

A correct BM25 implementation requires:

1. Building an inverted index that maps each unique token to the list of (document_id, term_frequency) pairs that contain it.
2. Computing and storing per-document lengths (|D|) and the average document length (avgdl) at index build time.
3. Computing document frequency (df) per term from the inverted index.
4. Tokenising queries identically to how documents were tokenised at index time.

## Limitations

- BM25 operates on exact token matches. Synonyms, morphological variants (run/running), and paraphrases do not contribute to the score unless the query and document share the exact same form.
- It is sensitive to the choice of tokeniser. Stop word removal and stemming can substantially affect scoring.
- It cannot generalise to zero-shot queries in languages or domains not represented in the training corpus.

## When to Use BM25

BM25 excels when:
- Query terms are highly specific and likely to appear verbatim in relevant documents.
- The corpus vocabulary is well-aligned with user query vocabulary.
- Low latency is required and dense retrieval compute is unavailable.

In hybrid retrieval systems, BM25 is typically combined with a dense retriever to capture both lexical and semantic relevance signals.
