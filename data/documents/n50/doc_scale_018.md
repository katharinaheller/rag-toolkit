DOCUMENT_ID: doc_scale_018
TITLE: Comparing BM25 with TF-IDF
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v8

CONTENT:
TF-IDF (Term Frequency–Inverse Document Frequency) is a predecessor to BM25
that shares the same intuition but lacks the saturation and length-normalisation
mechanisms that make BM25 more robust. In TF-IDF the term frequency component
grows linearly with the number of occurrences of a term, meaning documents that
repeat query terms many times receive disproportionately high scores regardless
of document length. BM25 replaces the linear TF with a logarithmic saturation
function, preventing this over-counting. TF-IDF also lacks a principled
document length normalisation; BM25's b parameter provides a tunable control
over this normalisation. Empirical benchmarks consistently show BM25 outperforming
TF-IDF across a wide variety of retrieval tasks, which explains why BM25 has
become the de-facto standard for sparse lexical retrieval.
