DOCUMENT_ID: doc_scale_026
TITLE: Matryoshka Representation Learning
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v6

CONTENT:
Matryoshka Representation Learning (MRL) is a training strategy that produces
embedding models where the first d dimensions of the full-length embedding
form a semantically valid representation for any supported dimensionality d.
This property allows a single embedding model to serve multiple deployment
scenarios: a high-accuracy scenario using the full 1024-dimensional embedding
and a low-latency or memory-constrained scenario using a truncated 256- or
128-dimensional embedding without any additional fine-tuning. MRL is
implemented by computing the contrastive loss at multiple embedding prefixes
simultaneously during training, forcing the model to pack the most important
information into the earliest dimensions. Models such as EmbeddingGemma
support MRL with output dimensions of 768, 512, 256, and 128, allowing
flexible trading off between embedding size, index memory, and retrieval
quality.
