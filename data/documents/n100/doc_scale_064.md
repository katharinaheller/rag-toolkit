DOCUMENT_ID: doc_scale_064
TITLE: Instruction-Tuned Embedding Models
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v4

CONTENT:
Instruction-tuned embedding models accept a natural language task description
alongside the input text, conditioning the encoder to produce embeddings
optimised for the specified task without further fine-tuning. The E5 family
of models (E5-base, E5-large, E5-mistral-7b) prepend a task prefix such as
"query: " or "passage: " to differentiate query and document representations.
The GTE (Generalised Text Embedding) models from Alibaba Cloud use similar
instruction conditioning. Instruction-tuned models consistently outperform
base models on the MTEB benchmark, particularly for tasks where the retrieval
objective differs from a generic semantic similarity objective. The practical
implication is that retrieval systems should select a model that is
instruction-tuned for the specific retrieval task (question-to-passage,
command-to-code, or document-to-document matching) rather than defaulting
to a general-purpose embedding model.
