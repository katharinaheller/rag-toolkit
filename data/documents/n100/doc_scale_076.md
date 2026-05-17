DOCUMENT_ID: doc_scale_076
TITLE: Repeat Penalty in LLM Inference
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v6

CONTENT:
The repeat penalty parameter discourages a language model from repeating
the same tokens or phrases within its generated output by downweighting the
logits of recently generated tokens. In Ollama and llama.cpp, the repeat
penalty is applied by multiplying the logit of each token that has appeared
in the recent generation history by 1 / repeat_penalty (for repeat_penalty
> 1.0), reducing its probability of being selected again. A repeat_penalty
of 1.0 applies no penalty (default behaviour), while values of 1.1 to 1.3
reduce repetition noticeably. For RAG generation tasks where the model must
closely follow the retrieved context, a moderate repeat penalty of 1.1 is
recommended to prevent excessive parroting of the context while avoiding
the distortion of the probability distribution that high penalties can cause.
