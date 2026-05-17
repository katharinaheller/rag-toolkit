DOCUMENT_ID: doc_scale_061
TITLE: Embedding Models for Text Retrieval
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v1

CONTENT:
Embedding models convert variable-length text into fixed-length dense vectors
that capture semantic meaning in a continuous vector space. In information
retrieval, embedding models serve as the backbone of dense retrieval systems
by encoding both queries and documents into vectors whose similarity can be
computed by dot product or cosine distance. The first generation of text
embedding models used static word vectors (Word2Vec, GloVe) that did not
capture context; modern models use transformer encoders that produce
contextualised representations sensitive to word order and surrounding context.
BERT-based bi-encoders, cross-encoders, and encoder-only models such as
BGE-M3 and E5 are the dominant architectures for text retrieval embedding.
The MTEB (Massive Text Embedding Benchmark) leaderboard tracks performance
across 58 datasets covering retrieval, clustering, classification, and
semantic similarity tasks.
