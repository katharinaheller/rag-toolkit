DOCUMENT_ID: doc_scale_039
TITLE: Brute-Force vs FAISS in RAG Evaluation
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v9

CONTENT:
During RAG system development, it is best practice to begin retrieval
evaluation using exact brute-force search rather than an approximate index.
This ensures that retrieval quality metrics reflect the true upper bound
achievable by the embedding model rather than being confounded by ANN recall
loss. Once the embedding model and top-k selection are validated on exact
search, the approximate index can be introduced and its recall penalty measured
by comparing nDCG@10 and MRR between the exact and approximate search variants.
A recall penalty below 2-3 percentage points is generally acceptable for
production deployment. If the penalty is higher, increasing nprobe (for IVF)
or M and efSearch (for HNSW) until acceptable recall is achieved without
exceeding latency budgets is the standard tuning procedure.
