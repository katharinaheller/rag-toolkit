DOCUMENT_ID: doc_scale_042
TITLE: Reciprocal Rank Fusion
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v2

CONTENT:
Reciprocal Rank Fusion (RRF) is a rank-based score fusion method that combines
multiple ranked lists into a single merged ranking without requiring score
normalisation. For each document, its RRF score is computed as the sum of
1 / (k + rank_r) over all ranking sources r, where rank_r is the document's
rank in source r and k is a smoothing constant that prevents very high scores
for top-ranked documents in a single list. The smoothing constant k is
typically set to 60 based on empirical experiments showing this value to be
robust across a wide variety of retrieval tasks. RRF has several attractive
properties: it is insensitive to the absolute scale of relevance scores from
different systems, does not require held-out data to calibrate, and is trivial
to implement. However, it treats all ranking sources as equally important;
when one source is substantially better than another, weighted alternatives
may outperform RRF.
