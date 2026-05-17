DOCUMENT_ID: doc_scale_001
TITLE: Retrieval-Augmented Generation: Core Concepts
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v1

CONTENT:
Retrieval-Augmented Generation (RAG) is a hybrid natural language processing
architecture that extends language models by grounding their responses in
external, domain-specific knowledge retrieved at inference time.
Introduced by Lewis et al. in 2020 at NeurIPS, RAG combines a parametric
model (the generator) with a non-parametric memory component (the retriever).
The generator is a sequence-to-sequence or decoder-only language model, while
the retriever selects relevant passages from a document corpus.
During inference the system first retrieves top-k passages relevant to the
user query, then conditions the generator on both the query and those passages
to produce the final answer. This design reduces hallucinations because the
generator is anchored to factual retrieved text rather than relying solely on
its compressed parametric knowledge. RAG systems are particularly effective for
knowledge-intensive tasks such as open-domain question answering, fact
verification, and enterprise search over proprietary documents.
