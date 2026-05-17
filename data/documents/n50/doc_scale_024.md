DOCUMENT_ID: doc_scale_024
TITLE: Training Dense Retrievers: Contrastive Learning
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v4

CONTENT:
Dense retrieval models are trained with contrastive objectives that pull
query-positive pairs together in embedding space and push query-negative pairs
apart. The standard approach uses in-batch negatives: for each query in a
training batch, all other queries' positive passages serve as negatives.
Hard negatives, retrieved by BM25 or a previous model checkpoint, are more
challenging than random negatives and consistently improve model quality.
The training loss is the negative log-likelihood of the positive passage score
relative to all passage scores in the batch. ANCE (Approximate Nearest
Neighbour Negative Contrastive Estimation) extends this by periodically mining
hard negatives from the current model's index, creating a curriculum-style
training loop that progressively refines the embedding space. Fine-tuning on
domain-specific pairs is critical for deployment in specialised domains such
as biomedical, legal, or technical documentation retrieval.
