DOCUMENT_ID: doc_scale_020
TITLE: BM25 Query Expansion Techniques
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v10

CONTENT:
Query expansion is a technique that augments the original query with related
terms to reduce vocabulary mismatch in BM25 retrieval. Pseudo-relevance
feedback (PRF) performs an initial retrieval pass, extracts terms from the
top-k retrieved documents, and adds high-weight terms back to the query for
a second retrieval pass. RM3 (Relevance Model 3) is a principled PRF method
that interpolates the original query model with an estimated relevance model
derived from the top documents. Alternatively, query expansion using external
knowledge bases such as WordNet or domain-specific thesauri adds synonyms
and related concepts directly to the query without requiring an initial
retrieval step. Large language models can also be used to generate paraphrases
of the query, which are then combined with BM25 to form a multi-representation
retrieval system.
