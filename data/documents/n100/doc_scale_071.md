DOCUMENT_ID: doc_scale_071
TITLE: Mistral 7B: Architecture and Performance
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v1

CONTENT:
Mistral 7B is a large language model with 7 billion parameters released by
Mistral AI in September 2023. Despite having fewer parameters than models
such as LLaMA-2 13B, Mistral 7B achieves comparable or superior performance
on a wide range of benchmarks including commonsense reasoning, code
generation, and reading comprehension. The model uses grouped-query attention
(GQA) for faster inference and reduced key-value cache memory, and sliding
window attention (SWA) to handle context windows up to 32768 tokens.
Mistral 7B is released under the Apache 2.0 licence, permitting commercial
use and fine-tuning. The model is available in instruction-tuned variants
(Mistral 7B Instruct) optimised for following natural language instructions
and generating helpful responses in a conversational format.
