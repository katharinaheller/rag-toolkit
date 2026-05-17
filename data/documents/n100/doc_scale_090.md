DOCUMENT_ID: doc_scale_090
TITLE: Faithfulness and Groundedness Metrics
CATEGORY: Evaluation
CORPUS_SIZES: n100
VARIANT: v10

CONTENT:
Faithfulness metrics quantify the degree to which each statement in the
generated answer is supported by the retrieved context, addressing the
hallucination concern in RAG systems. FactScore decomposes the generated
answer into atomic claims and verifies each claim against the retrieved
passages, yielding a faithfulness score between 0 and 1. AlignScore similarly
measures semantic alignment between generated sentences and context passages
using an NLI-based classifier. These metrics differ from generation quality
metrics such as Token F1: a faithful answer that is wrong (because the
retrieved context was wrong) receives a high faithfulness score but a low
Token F1. Conversely, a correct answer that was confabulated from parametric
knowledge rather than the retrieved context receives low faithfulness but
potentially high Token F1. Both types of metrics are necessary to fully
characterise RAG system behaviour.
