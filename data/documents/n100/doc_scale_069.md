DOCUMENT_ID: doc_scale_069
TITLE: Evaluating Embedding Models for Retrieval
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v9

CONTENT:
Selecting an embedding model for a new RAG deployment requires systematic
evaluation on a representative sample of the target queries and passages.
The MTEB benchmark provides a standardised evaluation framework but may not
reflect the specific distribution of the deployment corpus. A practical
evaluation workflow involves creating a small labelled evaluation set of
100-500 query-relevant-passage pairs from the target corpus, running each
candidate embedding model on this set, and comparing nDCG@10, MRR, and
Recall@10. The evaluation should also measure embedding throughput and
memory footprint to identify models that meet both quality and efficiency
constraints. If no single model dominates on all criteria, a hybrid approach
combining a fast small model for first-stage retrieval with a larger model
for re-ranking may achieve the best practical trade-off.
