DOCUMENT_ID: doc_scale_077
TITLE: Mistral vs LLaMA for RAG Applications
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v7

CONTENT:
For RAG applications running on CPU hardware, Mistral 7B Instruct is often
preferred over LLaMA-2 7B due to its superior instruction-following ability,
longer context window (32768 vs 4096 tokens), and comparable or better
performance on factual QA benchmarks. LLaMA-3 8B Instruct is a more recent
competitor with better overall benchmark performance and an improved
tokeniser. For GPU-rich deployments, larger models such as Mistral Nemo (12B)
or Mixtral 8x7B (mixture of experts) achieve substantially higher answer
quality for complex multi-hop questions. The correct model choice depends on
the deployment hardware budget, required answer quality, and acceptable
generation latency. A practical approach is to start with Mistral 7B Q4_K_M
for rapid prototyping and upgrade to a larger model when quality metrics
indicate the bottleneck is generation rather than retrieval.
