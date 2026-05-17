DOCUMENT_ID: doc_scale_096
TITLE: Ollama Model Management
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v6

CONTENT:
Managing models in Ollama involves pulling, listing, and deleting model
files through either the CLI or the REST API. The 'ollama pull' command
downloads a model from the Ollama library or a Hugging Face GGUF repository
and stores it in the local model registry at ~/.ollama/models. Model tags
follow the format 'name:tag', where the tag specifies the quantisation level
(q4_K_M, q5_K_M, q8_0) and optionally the parameter count for mixture-of-
experts models. The 'ollama list' command shows all locally available models
with their size and modification date. The 'ollama rm' command permanently
deletes a model from the registry. Custom models can be defined using a
Modelfile that specifies a base model, system prompt, and generation
parameters, allowing the creation of specialised model variants without
fine-tuning.
