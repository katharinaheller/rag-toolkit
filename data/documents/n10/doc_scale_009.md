DOCUMENT_ID: doc_scale_009
TITLE: Open-Domain Question Answering with RAG
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v9

CONTENT:
Open-domain question answering (ODQA) was one of the earliest and most
studied applications of Retrieval-Augmented Generation. In the ODQA setting,
the model must answer arbitrary factual questions without access to a pre-
computed answer database, relying instead on a large passage corpus such as
Wikipedia. The original RAG paper by Lewis et al. benchmarked on Natural
Questions, TriviaQA, and WebQuestions, demonstrating that a generator
conditioned on retrieved Wikipedia passages outperformed purely parametric
models of equivalent scale. The key insight was that retrieval allows the
model to access factual knowledge without memorising it, freeing model
capacity for reasoning and language generation rather than factual storage.
Modern ODQA systems extend this with multi-hop retrieval chains to answer
questions requiring information from multiple passages.
