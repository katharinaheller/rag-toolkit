DOCUMENT_ID: doc_scale_036
TITLE: Product Quantisation for Vector Compression
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v6

CONTENT:
Product Quantisation (PQ) is a vector compression technique used in FAISS to
reduce the memory footprint of large vector collections. PQ divides each D-
dimensional vector into M equal sub-vectors and quantises each sub-vector to
one of 256 cluster centroids (8-bit codes). The compressed representation
stores M bytes per vector instead of D * 4 bytes, achieving a compression
ratio of D / M * 4. For 1024-dimensional vectors with M=64, PQ reduces
memory from 4096 bytes to 64 bytes per vector — a 64× compression. Distance
computations in PQ space use precomputed lookup tables mapping code pairs
to approximate distances, enabling fast asymmetric distance computation (ADC).
The recall penalty of PQ depends on M and the number of centroids: more
sub-vectors and more centroids yield higher recall but larger codebook memory
and longer build times.
