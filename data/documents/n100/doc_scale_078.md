DOCUMENT_ID: doc_scale_078
TITLE: Top-p and Top-k Sampling in LLMs
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v8

CONTENT:
Nucleus sampling (top-p) and top-k sampling are techniques that restrict
the vocabulary to a subset of tokens before sampling the next output token.
Top-k sampling selects the k highest-probability tokens and samples from
that restricted vocabulary, where k is typically 40-100. Top-p (nucleus)
sampling selects the smallest set of tokens whose cumulative probability
exceeds p (typically 0.9 or 0.95) and samples from that set. Nucleus sampling
adapts the effective vocabulary size to the model's confidence: when the
model is confident, the nucleus may contain only 2-3 tokens; when uncertain,
it may contain hundreds. For RAG generation with temperature 0.0, top-p and
top-k parameters have no effect because greedy decoding bypasses sampling
entirely. They become relevant for non-zero temperature settings used in
creative or diverse generation scenarios.
