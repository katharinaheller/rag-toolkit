DOCUMENT_ID: doc_scale_062
TITLE: EmbeddingGemma: Gemma-Based Embedding
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v2

CONTENT:
EmbeddingGemma is an embedding model derived from the Gemma decoder-only
large language model architecture. Unlike BERT-style encoder-only models,
EmbeddingGemma uses the last token's representation (or a pooled representation)
of a decoder model as the document embedding. Its native embedding dimension
is 768, and it supports Matryoshka Representation Learning with truncated
dimensions of 512, 256, and 128. A critical implementation detail is that
EmbeddingGemma must not be loaded with float16 precision because the Gemma
architecture's numerics produce NaN or overflow values when run in half-
precision, unlike encoder-only models such as BERT or BGE-M3 that handle
float16 well. For GPU inference, bfloat16 (use_bfloat16=True) is the correct
reduced-precision format for EmbeddingGemma, providing memory savings without
numerical instability.
