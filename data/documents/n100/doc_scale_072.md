DOCUMENT_ID: doc_scale_072
TITLE: Quantised Inference: Q4_K_M and GGUF
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v2

CONTENT:
GGUF (GPT-Generated Unified Format) is a file format for storing quantised
large language model weights designed for CPU inference using the llama.cpp
library. Quantisation reduces the precision of model weights from float32 or
bfloat16 to lower-bit representations such as 4-bit, 5-bit, or 8-bit,
significantly reducing memory requirements and improving inference speed on
CPU hardware at a small cost to model quality. The Q4_K_M quantisation format
uses 4-bit weights with a K-quantisation scheme that applies higher precision
to the most sensitive weight matrices in the attention mechanism. For Mistral
7B, Q4_K_M reduces the model size from approximately 14 GB (bfloat16) to
approximately 4.1 GB, enabling inference on consumer hardware with 8-16 GB
of RAM. On CPU, Mistral 7B Q4_K_M generates approximately 3-8 tokens per
second depending on hardware, compared to 30-50 tokens per second on a
modern GPU.
