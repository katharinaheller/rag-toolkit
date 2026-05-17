DOCUMENT_ID: doc_scale_052
TITLE: Fixed-Size Character Chunking with Overlap
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v2

CONTENT:
Fixed-size character chunking splits documents at fixed character boundaries
with a configurable overlap. Given a chunk size of C characters and an overlap
of O characters, consecutive chunks share O characters of text, preventing
relevant information near chunk boundaries from being split in half. A
starting point for most RAG applications is chunk_size=512 characters with
overlap=64 characters. For dense technical documentation where each paragraph
is largely self-contained, chunk_size=256 with overlap=32 may produce higher
retrieval precision. For narrative or conversational text where context spans
multiple paragraphs, larger chunks of 1024 or 2048 characters better preserve
coherence. Character-level chunking is simple to implement and language-
agnostic, but splits may fall in the middle of sentences, disrupting
readability and embedding quality.
