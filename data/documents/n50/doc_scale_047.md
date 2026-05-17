DOCUMENT_ID: doc_scale_047
TITLE: Query-Type Adaptation in Hybrid Retrieval
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v7

CONTENT:
Different query types benefit from different retrieval strategies, which
motivates adaptive or query-type-conditioned hybrid retrieval. Factoid
questions with named entities (product codes, person names, technical terms)
tend to benefit from a higher sparse weight because BM25 exactly matches
rare entity strings. Conceptual or semantic queries (asking for explanations,
comparisons, or summaries of concepts) tend to benefit from a higher dense
weight because the relevant passages may not share exact lexical overlap with
the query. Automatic query type classification using a lightweight classifier
or rule-based heuristics can route queries to the appropriate fusion strategy
at inference time. Self-attention head analysis shows that dense encoders
attend to semantic relationships while sparse representations highlight
discriminative keywords, which aligns with the intuition underlying
adaptive fusion.
