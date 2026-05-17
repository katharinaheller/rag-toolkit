DOCUMENT_ID: doc_scale_079
TITLE: Hallucination in LLMs and RAG Mitigation
CATEGORY: Language Models
CORPUS_SIZES: n100
VARIANT: v9

CONTENT:
Hallucination refers to the tendency of large language models to generate
fluent, confident-sounding text that is factually incorrect or unsupported
by any training data. In RAG systems, hallucination manifests as the model
generating answers that contradict or go beyond the retrieved context, drawing
on its parametric knowledge rather than the grounding passages. Mitigation
strategies include strict RAG prompts that explicitly forbid generating
information not present in the context, answer verification steps that check
whether the generated answer can be textually entailed by the retrieved
passages, and ensemble generation that produces multiple candidates and selects
the one best supported by the retrieved context. Faithfulness metrics such as
FactScore and AlignScore measure the degree to which generated statements are
supported by the retrieved passages, providing an automated hallucination
detection signal for monitoring in production.
