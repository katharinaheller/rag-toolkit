DOCUMENT_ID: doc_scale_012
TITLE: BM25 Hyperparameter Tuning: k1 and b
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v2

CONTENT:
The two primary hyperparameters of BM25 control its sensitivity to term
frequency saturation and document length normalisation. The parameter k1,
typically set between 1.2 and 2.0, governs how quickly term frequency
influence saturates as repeated occurrences of a query term appear in a
document. A higher k1 allows more differentiation between low-frequency and
high-frequency documents, while a lower k1 approaches a binary term presence
model. The parameter b, usually set to 0.75, controls the degree of document
length normalisation relative to the average document length in the corpus.
Setting b to 1.0 produces full length normalisation (longer documents are
penalised proportionally), while b = 0.0 disables normalisation entirely.
Optimal values for k1 and b depend on the corpus and query distribution and
are typically found through grid search on a validation query set.
