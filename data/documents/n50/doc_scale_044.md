DOCUMENT_ID: doc_scale_044
TITLE: Learned Sparse Retrieval: SPLADE
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v4

CONTENT:
Learned sparse retrieval models such as SPLADE (Sparse Lexical and Expansion
model) bridge the gap between BM25 and dense retrieval by producing sparse
vectors over the full vocabulary rather than binary term presence indicators.
SPLADE uses a masked language model head to predict a positive weight for
every vocabulary token given the input document, producing a high-dimensional
sparse representation where non-zero weights indicate semantically relevant
terms even if they do not appear literally in the document. This enables
semantic expansion while retaining the inverted-index infrastructure that makes
BM25 scalable. BGE-M3 implements a variant of learned sparse retrieval using
its sparse encoder head, producing vocabulary-sized sparse vectors that can
be indexed in an inverted index and scored efficiently at query time using dot
product over the sparse representations.
