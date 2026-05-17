DOCUMENT_ID: doc_scale_008
TITLE: RAG Evaluation Challenges
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v8

CONTENT:
Evaluating a RAG system requires measuring both retrieval quality and
generation quality, since each stage can introduce independent error. Retrieval
quality metrics such as Recall@k and nDCG@k assess whether the relevant
passages appear in the retrieved set. Generation quality metrics such as
Exact Match, Token F1, and ROUGE measure how well the generated answer matches
a reference. End-to-end metrics combine both views but obscure the source of
failure. LLM-as-a-judge evaluation, in which a separate language model rates
the generated answer, provides richer feedback but introduces a dependency on
the judge model's biases and API costs. RAGAS and similar frameworks attempt
to automate RAG evaluation through a combination of faithfulness, answer
relevance, and context precision scores, though they require their own
validation before being trusted in a production evaluation harness.
