DOCUMENT_ID: doc_scale_004
TITLE: Naive vs Advanced RAG Variants
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v4

CONTENT:
The literature distinguishes several generations of RAG system design.
Naive RAG, the original formulation, performs a single retrieval pass before
generation with no iterative refinement. Advanced RAG introduces pre-retrieval
and post-retrieval enhancements: query rewriting, hypothetical document
embeddings (HyDE), and re-ranking of retrieved passages before prompting the
generator. Modular RAG decouples each pipeline stage into independently
replaceable components, allowing practitioners to swap retrievers, re-rankers,
and generators without touching the rest of the system. Iterative RAG further
extends this by allowing the generator to issue follow-up retrieval requests
when the initial context is insufficient to answer multi-hop questions.
Choosing between these variants involves balancing accuracy, latency, and
system complexity for the target deployment scenario.
