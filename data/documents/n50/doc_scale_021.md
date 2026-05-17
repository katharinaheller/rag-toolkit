DOCUMENT_ID: doc_scale_021
TITLE: Dense Passage Retrieval: DPR
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v1

CONTENT:
Dense Passage Retrieval (DPR) is a bi-encoder framework that maps queries
and passages into a shared dense vector space such that relevant passage
vectors are close to the query vector under an inner-product or cosine
similarity measure. Introduced by Karpukhin et al. in 2020, DPR uses two
independent BERT-based encoders: a query encoder and a passage encoder.
The passage encoder is applied offline to all passages in the corpus and the
resulting embeddings are stored in a FAISS index. At query time the query
encoder produces a query embedding, and a maximum inner-product search
retrieves the top-k passages with the highest similarity scores. DPR trained
with in-batch negatives on Natural Questions-style training data substantially
outperformed BM25 on open-domain question answering benchmarks, establishing
dense retrieval as a viable alternative to sparse lexical retrieval.
