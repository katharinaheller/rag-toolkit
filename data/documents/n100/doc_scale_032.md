DOCUMENT_ID: doc_scale_032
TITLE: FAISS IndexFlat: Exact Brute-Force Search
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v2

CONTENT:
IndexFlat is the simplest FAISS index type: it stores all vectors explicitly
and performs exact brute-force search by computing the full inner product or
L2 distance between the query vector and every stored vector. IndexFlatL2
uses squared L2 distance and is appropriate for unnormalised embeddings,
while IndexFlatIP uses inner product and is appropriate for normalised
embeddings (where it is equivalent to cosine similarity). IndexFlat guarantees
100% recall because it exhaustively searches all vectors, making it the
correct baseline for evaluating approximate indexes. The memory requirement
is proportional to N * D * 4 bytes for float32 vectors, where N is the number
of vectors and D is the dimensionality. For one million 1024-dimensional
float32 vectors, IndexFlat requires approximately 4 GB of RAM, which is
manageable for small-to-medium corpora but prohibitive at billion-scale.
