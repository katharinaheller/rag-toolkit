DOCUMENT_ID: doc_scale_029
TITLE: Embedding Normalisation and Similarity Metrics
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v9

CONTENT:
The choice of similarity metric and embedding normalisation strategy has a
significant effect on dense retrieval quality. L2-normalised embeddings with
cosine similarity and raw embeddings with inner-product (dot-product) similarity
are the two dominant choices. After L2 normalisation, cosine similarity and
dot-product are equivalent, so the distinction collapses to whether
normalisation is applied. Some models are trained with normalisation applied
during training, and those models should be normalised at inference time to
remain consistent with the training distribution. BGE-M3 is trained with
normalisation and benefits from applying L2 normalisation before indexing.
Asymmetric retrieval, in which the query encoder and document encoder have
different architectures or normalisation schemes, is used when query and
document length distributions differ significantly.
