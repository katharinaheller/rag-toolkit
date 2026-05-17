DOCUMENT_ID: doc_scale_016
TITLE: BM25+ and BM25F Variants
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v6

CONTENT:
Several variants of the original BM25 formula have been proposed to address
specific limitations. BM25+ modifies the term frequency normalisation to
assign a minimum non-zero score even to very long documents, correcting an
over-penalisation bug in the original formulation for documents that are
much longer than average. BM25F (Fields) extends the model to structured
documents by computing separate term frequency values for different document
fields such as title, abstract, and body, and combining them with field-
specific weights before applying IDF. This allows the retrieval system to
give higher importance to query terms appearing in the title than in the body
of a document, matching common relevance intuitions. BM25F has been shown to
significantly improve retrieval precision on structured web documents.
