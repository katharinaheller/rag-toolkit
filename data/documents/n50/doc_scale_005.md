DOCUMENT_ID: doc_scale_005
TITLE: RAG Limitations and Failure Modes
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v5

CONTENT:
Despite its advantages, Retrieval-Augmented Generation is not free from
failure modes. Retrieval failures occur when the most relevant passages rank
below the top-k threshold, leaving the generator without the information needed
to answer correctly. Context overload arises when too many retrieved passages
compete for the model's attention window, diluting the signal. Faithfulness
errors happen when the generator ignores the retrieved context and falls back
on its parametric memory, producing confident but unsupported answers.
Chunk boundary effects cause relevant information to be split across passages,
reducing the coherence of the retrieved context. Finally, distributional
mismatch between the query encoder and the document encoder leads to poor
embedding alignment. Mitigating these failure modes requires careful tuning
of chunk size, top-k, re-ranking strategies, and prompt construction.
