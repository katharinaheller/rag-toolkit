DOCUMENT_ID: doc_scale_087
TITLE: ROUGE and BERTScore for Generation Quality
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v7

CONTENT:
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a family of
n-gram overlap metrics commonly used for evaluating text summarisation and
machine translation. ROUGE-L measures the longest common subsequence between
the generated and reference text, providing a more lenient match than exact
n-gram comparison. BERTScore computes similarity between generated and
reference tokens using contextualised BERT embeddings, capturing semantic
equivalences that n-gram overlap misses. In RAG evaluation, ROUGE and
BERTScore are alternatives to Token F1 when the reference answers are longer
or when paraphrase handling is important. However, these metrics require
reference answers, which may not be available in open-ended generation
settings. For RAG systems evaluated on factual question answering, Token F1
remains the standard metric due to its simplicity and correlation with human
judgements.
