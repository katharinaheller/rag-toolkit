DOCUMENT_ID: doc_scale_013
TITLE: BM25 Inverted Index Construction
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v3

CONTENT:
BM25 retrieval is powered by an inverted index that maps each term in the
vocabulary to a posting list of document identifiers and per-document term
frequencies. Building this index involves tokenising each document, computing
term frequencies, and appending entries to the relevant posting lists.
At query time the query terms are looked up in the index, their posting lists
are merged, and a BM25 score is computed for each candidate document by
combining the pre-computed IDF values with the stored term frequencies.
The inverted index supports very fast query processing because it skips
documents that do not contain any query term, focusing computation only on
candidates in the union of posting lists. Modern implementations use
compression schemes such as variable-byte encoding or SIMD-accelerated
block decompression to minimise both storage and query latency.
