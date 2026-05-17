DOCUMENT_ID: doc_scale_007
TITLE: Prompt Construction in RAG Systems
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v7

CONTENT:
The way retrieved context is formatted into a prompt has a large influence on
generation quality in RAG systems. A strict RAG prompt places a clear
instruction such as "Answer using only the context below" ahead of the
retrieved passages, reducing hallucination at the cost of flexibility.
A chain-of-thought RAG prompt asks the model to reason step-by-step before
providing its final answer, which improves performance on complex multi-step
questions. Context ordering experiments have shown that placing the most
relevant passage first, rather than in arbitrary order, improves the quality
of the final answer due to primacy and recency effects in attention.
Truncating context to fit within the model's context window requires a
priority scheme, such as preferring higher-ranked passages and dropping lower-
ranked ones when the budget is exhausted.
