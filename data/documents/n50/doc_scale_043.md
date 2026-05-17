DOCUMENT_ID: doc_scale_043
TITLE: Weighted Score Fusion for Hybrid Retrieval
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v3

CONTENT:
Weighted score fusion combines the normalised relevance scores from sparse and
dense retrieval using a convex combination: hybrid_score = alpha * dense_score
+ (1 - alpha) * sparse_score. The weights alpha and (1 - alpha) are calibrated
on a validation query set to reflect the relative quality of each retrieval
system on the target domain. Unlike RRF, weighted fusion requires score
normalisation: scores from each system must be mapped to the same [0, 1]
range before combination. Min-max normalisation across the retrieved set or
softmax normalisation are common choices. A limitation of weighted fusion is
that the optimal alpha varies across query types: factual lookup queries may
benefit from higher sparse weight while conceptual paraphrase queries may
benefit from higher dense weight. Learned query-type classifiers that
dynamically adjust alpha per query have been proposed to address this.
