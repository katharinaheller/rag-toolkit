DOCUMENT_ID: doc_scale_023
TITLE: Bi-Encoder vs Cross-Encoder Architectures
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v3

CONTENT:
Dense retrieval models fall into two architectural families: bi-encoders and
cross-encoders. Bi-encoders process the query and the document independently,
producing separate embeddings whose similarity is computed by a dot product or
cosine function. This architecture allows documents to be pre-encoded offline,
enabling efficient inference over millions of passages. Cross-encoders process
the query and document jointly through a single transformer, allowing full
attention across both inputs and typically producing higher-quality relevance
scores. However, cross-encoders cannot pre-compute document representations,
making them too slow for first-stage retrieval over large corpora. In practice,
retrieval systems use bi-encoders for the first-stage candidate selection and
cross-encoders as re-rankers over the short-listed candidate set, balancing
efficiency and accuracy.
