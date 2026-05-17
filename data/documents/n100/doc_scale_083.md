DOCUMENT_ID: doc_scale_083
TITLE: nDCG: Normalised Discounted Cumulative Gain
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v3

CONTENT:
Normalised Discounted Cumulative Gain (nDCG) is a rank-aware retrieval metric
that rewards systems that place highly relevant documents at the top of the
ranked list. DCG at position k is computed as the sum of relevance scores
discounted by the logarithm of the rank position: DCG@k = sum_{i=1}^{k}
rel_i / log2(i+1). This logarithmic discounting reflects the intuition that
users are less likely to examine documents at lower ranks. nDCG normalises DCG
by the ideal DCG (IDCG), the maximum achievable DCG if all relevant documents
were placed at the top. nDCG values range from 0 to 1, with 1 indicating a
perfect ranking. nDCG supports graded relevance judgements (relevance scores
of 0, 1, 2, or 3) as well as binary relevance (0 or 1), making it more
expressive than MRR for tasks with multiple degrees of relevance.
