DOCUMENT_ID: doc_scale_075
TITLE: Context Window Management in LLMs
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v5

CONTENT:
Mistral 7B supports a context window of 32768 tokens using sliding window
attention, compared to the 4096-token window of the original LLaMA models.
In RAG systems, the context window budget is allocated between the system
prompt, the retrieved passages, and the query, with the generator's response
consuming additional tokens. Typical RAG prompts use 500-2000 tokens for the
context, leaving ample headroom within the 32768-token window. However,
models exhibit a "lost in the middle" effect where information placed in the
middle of a long context is attended to less effectively than information at
the beginning or end. Placing the most relevant retrieved passage first in
the prompt context is a simple heuristic that mitigates this effect. For
very long contexts, recursive summarisation or map-reduce generation patterns
can process documents that exceed the model's window.
