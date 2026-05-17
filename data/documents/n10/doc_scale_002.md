DOCUMENT_ID: doc_scale_002
TITLE: RAG Architecture: Components and Data Flow
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v2

CONTENT:
A Retrieval-Augmented Generation pipeline consists of five primary stages
that transform a raw user query into a grounded natural language response.
In the first stage, ingestion, source documents are chunked into overlapping
text segments and stored in a document corpus. The second stage, embedding,
converts each chunk into a dense vector representation using an encoder model.
Third, indexing organises those vectors into a data structure that supports
efficient approximate nearest-neighbour search. Fourth, retrieval accepts the
query, embeds it using the same encoder, and searches the index to recover the
top-k most relevant chunks. Fifth, generation concatenates the retrieved chunks
with the original query into a prompt that a large language model uses to
synthesise a coherent, grounded answer. Each stage can be configured and
optimised independently, making the architecture highly modular.
