DOCUMENT_ID: doc_scale_049
TITLE: Union vs Intersection Candidate Sets in Hybrid Retrieval
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v9

CONTENT:
When combining candidate sets from sparse and dense retrieval, two strategies
are possible: union and intersection. The union strategy includes all documents
returned by either retriever and scores documents present in only one list
using a zero or minimal score for the absent retriever. This strategy maximises
recall but can introduce low-quality candidates from each retriever that the
other has correctly excluded. The intersection strategy includes only documents
returned by both retrievers and assigns a combined score only to those. This
improves precision but reduces recall, potentially missing relevant documents
that are retrieved by only one system. RRF naturally implements the union
strategy by assigning a score of 1 / (k + max_rank) to documents absent from
a given list. Most production hybrid retrieval systems use the union strategy
with RRF because recall is more valuable than precision at the first-stage
retrieval step.
