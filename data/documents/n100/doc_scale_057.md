DOCUMENT_ID: doc_scale_057
TITLE: Chunk Size and Retrieval Precision
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v7

CONTENT:
Chunk size has a fundamental influence on retrieval precision and recall.
Smaller chunks contain less noise and their embeddings are more focused,
allowing the retrieval system to pinpoint the exact passage relevant to a
narrow query. However, very small chunks lose context, and each chunk may
lack sufficient information to answer a query on its own even if it is
technically retrieved. Larger chunks preserve more context per retrieval
result but reduce the discriminative ability of the embedding, mixing
information about multiple sub-topics into a single vector. Evaluation studies
recommend exploring chunk sizes from 128 to 1024 characters (or tokens) and
selecting based on the target nDCG@10 and Token F1 metrics on a held-out
evaluation set specific to the deployment domain.
