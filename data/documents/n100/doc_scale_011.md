DOCUMENT_ID: doc_scale_011
TITLE: BM25: Best Match 25 Algorithm
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v1

CONTENT:
BM25 (Best Match 25) is a probabilistic bag-of-words ranking function that
scores documents by their relevance to a query based on term frequency and
inverse document frequency. Developed within the Okapi project at City
University of London during the 1990s, BM25 remains one of the most widely
deployed information retrieval algorithms in production search systems.
The BM25 score for a document D given query Q is computed as the sum over
query terms of IDF(t) multiplied by a normalised term frequency factor.
The normalised TF factor is TF(t,D) * (k1 + 1) / (TF(t,D) + k1 * (1 - b +
b * |D| / avgdl)), where k1 and b are tunable hyperparameters. This
saturation function prevents very high-frequency terms from dominating the
score, which is a key advantage over earlier TF-IDF variants.
