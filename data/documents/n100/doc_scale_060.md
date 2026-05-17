DOCUMENT_ID: doc_scale_060
TITLE: Chunk Deduplication and Quality Filtering
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v10

CONTENT:
After chunking, it is important to deduplicate near-identical chunks and
filter low-quality chunks before indexing. Exact duplicate chunks arise when
the same boilerplate text (headers, footers, disclaimers) appears in many
source documents. Near-duplicate chunks can be identified using MinHash
LSH with a Jaccard similarity threshold or by computing the cosine similarity
of chunk embeddings and removing pairs above a threshold. Low-quality chunk
filtering removes chunks that are too short to contain meaningful information
(typically fewer than 50 characters), chunks consisting primarily of numbers
or special characters, and chunks that are primarily navigation elements
or copyright notices. A clean, deduplicated corpus reduces index size,
improves retrieval precision, and prevents the retrieval system from returning
multiple near-identical passages in the top-k results.
