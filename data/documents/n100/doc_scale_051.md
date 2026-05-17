DOCUMENT_ID: doc_scale_051
TITLE: Document Chunking: Motivation and Design
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v1

CONTENT:
Document chunking divides long source documents into smaller text segments
that fit within the context window of the embedding model and the language
model generator. Without chunking, a multi-page document would either exceed
the model's maximum input length or overwhelm the model's attention with
irrelevant content. Chunking also enables more precise retrieval: a query
about a specific subsection of a report should retrieve that subsection,
not the entire report. The ideal chunk contains enough context for a reader
to understand the passage in isolation and is short enough that the embedding
faithfully captures its main semantic content. Chunk size and chunk overlap
are the primary design parameters, and their optimal values depend on the
document type, query distribution, and embedding model characteristics.
