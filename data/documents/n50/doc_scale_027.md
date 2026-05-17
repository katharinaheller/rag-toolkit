DOCUMENT_ID: doc_scale_027
TITLE: Late Interaction: ColBERT and MaxSim
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v7

CONTENT:
ColBERT (Contextualised Late Interaction over BERT) is a dense retrieval model
that produces one embedding vector per token rather than a single document-level
vector. At retrieval time, the query-document relevance score is computed as
the sum of maximum similarities (MaxSim) between each query token embedding
and the set of document token embeddings. This late interaction mechanism
captures fine-grained token-level alignment while still allowing documents to
be encoded offline, giving ColBERT efficiency closer to bi-encoders than
cross-encoders. The trade-off is storage cost: storing one vector per token
requires 30-100x more storage than single-vector models. ColBERTv2 reduces
this overhead through vector quantisation and compression, making the approach
more practical for large corpora. BGE-M3 also supports multi-vector retrieval
as one of its three retrieval modes.
