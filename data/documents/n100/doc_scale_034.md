DOCUMENT_ID: doc_scale_034
TITLE: FAISS IVF: Inverted File Index
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v4

CONTENT:
The Inverted File (IVF) index in FAISS partitions the vector space into nlist
clusters using k-means clustering, then stores each vector in the posting list
of its nearest cluster centroid. At query time, the index identifies the
nprobe nearest centroids to the query vector and searches only those posting
lists, dramatically reducing the number of distance computations. IVF requires
a training phase using a representative sample of vectors to learn the cluster
centroids; a common rule of thumb is to use at least 10 * nlist training
vectors. With nprobe=1 only a single cluster is searched (fastest but lowest
recall), while nprobe=nlist degrades to brute-force search (100% recall but
no speedup). IVF is typically combined with Product Quantisation (IVF-PQ) to
further reduce memory by compressing residual vectors after cluster assignment.
