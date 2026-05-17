DOCUMENT_ID: doc_scale_054
TITLE: Semantic Chunking
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v4

CONTENT:
Semantic chunking segments documents at points where the semantic content
shifts significantly, as detected by measuring the cosine distance between
consecutive sentence embeddings. When the embedding similarity between two
adjacent sentences drops below a threshold, a chunk boundary is inserted.
This produces chunks that are semantically coherent rather than mechanically
sized, ensuring that a single chunk does not span multiple unrelated topics.
The implementation requires embedding all sentences in the document before
applying the segmentation algorithm, making it substantially more
computationally expensive than fixed-size or sentence-aware chunking.
Semantic chunking tends to produce variable-length chunks, which can be
padded or truncated to fit model context windows. Empirical evaluations show
that semantic chunking improves retrieval precision on multi-topic documents
such as research papers and technical reports but provides marginal benefit
for homogeneous single-topic documents.
