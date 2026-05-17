# Retrieval-Augmented Generation (RAG)

## Definition

Retrieval-Augmented Generation, commonly abbreviated as RAG, is a hybrid natural language processing architecture that augments the capabilities of large language models by grounding them in external, domain-specific knowledge retrieved at inference time. The term was introduced by Lewis et al. in their 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", published at NeurIPS 2020.

## Core Components

A RAG system consists of two primary components:

1. **Retriever**: A search system that, given a natural language query, identifies and returns the most relevant passages from a document corpus. The retriever may operate over dense vector representations, sparse lexical signals, or a combination of both.

2. **Generator**: A sequence-to-sequence language model that synthesises an answer by conditioning on both the query and the retrieved passages. The generator is typically a pre-trained transformer model fine-tuned for question answering or instruction following.

## Motivation

Large language models trained on static corpora cannot access information that postdates their training cutoff, cannot be updated without costly retraining, and frequently hallucinate facts that are not supported by evidence. RAG addresses these limitations by:

- Providing the model with relevant, up-to-date source documents at query time.
- Grounding generated responses in verifiable retrieved passages.
- Enabling factual answers about domain-specific knowledge without fine-tuning the model weights.

## Data Flow

The canonical RAG pipeline operates as follows:

1. The user submits a natural language query.
2. The query is encoded into a representation suitable for the retriever (e.g., a dense embedding vector or a list of query tokens).
3. The retriever searches the indexed document corpus and returns the top-k most relevant passages.
4. The retrieved passages, together with the original query, are assembled into a prompt.
5. The language model generates an answer conditioned on the prompt.
6. The answer is returned to the user.

## Variants

Several RAG variants have been proposed since the original work:

- **Naive RAG**: Single retrieval step followed by generation.
- **Advanced RAG**: Adds query rewriting, re-ranking, and multi-hop retrieval.
- **Modular RAG**: Treats each pipeline component as a replaceable module.
- **Iterative RAG**: Performs retrieval and generation in alternating steps.

## Advantages Over Pure LLM Inference

| Aspect | Pure LLM | RAG |
|--------|----------|-----|
| Domain-specific accuracy | Limited by training data | High, if corpus is curated |
| Hallucination risk | High | Reduced |
| Updatability | Requires retraining | Update the document corpus |
| Computational cost | Low (inference only) | Moderate (retrieval + inference) |
| Interpretability | Low | Higher (can cite sources) |

## Limitations

RAG is not a universal solution. Its effectiveness depends on:

- The quality and coverage of the document corpus.
- The relevance of retrieved passages to the query.
- The ability of the generator to correctly synthesise information from noisy or incomplete context.
- The computational cost of retrieval, especially for large corpora.
