DOCUMENT_ID: doc_scale_022
TITLE: BGE-M3: Multi-Functionality Embedding Model
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v2

CONTENT:
BGE-M3 (BAAI General Embedding, Multi-Functionality, Multi-Linguality, Multi-
Granularity) is an embedding model released by the Beijing Academy of Artificial
Intelligence that supports dense, sparse, and multi-vector (ColBERT-style)
retrieval from a single encoder. Its native dense embedding dimension is 1024,
and it supports input sequences of up to 8192 tokens, making it suitable for
long-document retrieval. On GPU inference, loading the model weights in
float16 (use_fp16=True) halves memory consumption without meaningful quality
loss. On CPU, float32 is recommended because float16 arithmetic is not
natively accelerated on most x86 processors. BGE-M3 trained on over 1 billion
pairs from diverse multilingual sources and demonstrates strong zero-shot
cross-lingual transfer across 100+ languages, making it a leading choice for
multilingual retrieval applications.
