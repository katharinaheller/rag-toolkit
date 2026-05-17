# Introduction

Retrieval-Augmented Generation (RAG) is a hybrid approach combining
information retrieval with generative language models.

## Components

A RAG system has three core components:

- **Retriever** — finds relevant documents.
- **Generator** — composes a final answer.
- **Index** — supports fast lookup over a corpus.

## Example

```python
def retrieve(query: str) -> list[str]:
    return search_index(query)
```

> RAG grounds the language model in concrete evidence,
> reducing hallucinations.

---

## Conclusion

The combination of retrieval and generation has become the
dominant pattern for question answering over private corpora.
