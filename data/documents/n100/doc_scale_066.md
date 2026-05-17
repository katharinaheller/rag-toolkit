DOCUMENT_ID: doc_scale_066
TITLE: Embedding Dimensionality and Retrieval Quality
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v6

CONTENT:
Higher embedding dimensionality generally correlates with better retrieval
quality because the model has more capacity to encode semantic distinctions.
However, dimensionality also directly impacts index memory (4 bytes per
dimension per vector for float32), query latency (dot product cost scales
linearly with dimension), and storage costs. The relationship between
dimensionality and quality is sub-linear: a 1024-dimensional model typically
outperforms a 256-dimensional model by only a few nDCG@10 points despite
having 4x more dimensions. Matryoshka models allow practitioners to explore
this trade-off by evaluating retrieval quality at multiple dimensionalities
from the same model, choosing the smallest dimension that meets the quality
threshold. For resource-constrained deployments, 256 or 384 dimensions
often provide 90-95% of the quality of the full model dimension at a
fraction of the cost.
