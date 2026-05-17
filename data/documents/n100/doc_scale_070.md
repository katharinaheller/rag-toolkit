DOCUMENT_ID: doc_scale_070
TITLE: Pooling Strategies for Transformer Embeddings
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v10

CONTENT:
Transformer models produce per-token hidden states as output, and a pooling
strategy must be applied to derive a single fixed-length embedding from these
token representations. The three main pooling strategies are CLS token pooling
(using the [CLS] token representation), mean pooling (averaging all token
representations), and max pooling (taking the element-wise maximum across all
token representations). CLS pooling was the original approach in BERT-based
models, where the CLS token was designed to aggregate sequence-level information
for classification tasks. Mean pooling is empirically superior for semantic
similarity and retrieval tasks because it incorporates information from all
tokens, including those at the end of the sequence. BGE-M3 uses mean pooling
for its dense embedding output. The choice of pooling strategy is a
hyperparameter that should be validated against the target task's evaluation
metric.
