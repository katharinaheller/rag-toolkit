DOCUMENT_ID: doc_scale_045
TITLE: Hybrid Retrieval Evaluation Strategies
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v5

CONTENT:
Evaluating hybrid retrieval systems requires controlled experiments that
isolate the contribution of each retrieval component. The standard procedure
starts by evaluating pure BM25 and pure dense retrieval independently on the
target benchmark, then evaluating the hybrid combination. If the hybrid
improvement over the best single-system baseline is less than 1-2 nDCG@10
points, the added complexity of maintaining two retrieval pipelines may not
be justified. Ablation experiments should vary the fusion weights (or k for
RRF) to characterise the sensitivity of the hybrid performance to these
hyperparameters. Query-level analysis comparing the set of queries where
hybrid outperforms each individual system reveals the qualitative patterns
that motivate the hybrid design and provides material for thesis discussion
sections on retrieval strategy selection.
