DOCUMENT_ID: doc_scale_081
TITLE: Retrieval Evaluation: Precision and Recall
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v1

CONTENT:
Context Precision measures the fraction of retrieved documents that are
relevant to the query: it is the number of relevant documents in the top-k
divided by k. High precision means that most retrieved documents are useful,
minimising noise passed to the generator. Context Recall measures the fraction
of all relevant documents in the corpus that appear in the top-k results:
it is the number of relevant documents in the top-k divided by the total
number of relevant documents. High recall ensures that no important information
is missed by the retriever. Precision and recall are complementary metrics;
increasing top-k typically increases recall at the cost of precision.
The harmonic mean of precision and recall, the F1 score, provides a single
balanced metric that is useful for comparing retrieval configurations.
