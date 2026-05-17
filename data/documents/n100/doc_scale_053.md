DOCUMENT_ID: doc_scale_053
TITLE: Sentence-Aware Chunking
CATEGORY: Chunking
CORPUS_SIZES: n100
VARIANT: v3

CONTENT:
Sentence-aware chunking uses a sentence boundary detector (such as a regex
pattern, spaCy sentence tokeniser, or NLTK punkt tokeniser) to identify
sentence boundaries before grouping sentences into chunks. This avoids
cutting through the middle of sentences, producing chunks that begin and end
at grammatically complete units. Sentence-aware chunking typically produces
more coherent embeddings than character-level chunking because the embedding
model sees complete linguistic units rather than truncated sentences. The
trade-off is computational cost: sentence tokenisation adds overhead to the
ingestion pipeline, particularly for very large corpora. A common
configuration groups 3-5 sentences per chunk with a one-sentence overlap.
For structured technical documents, paragraph boundaries often serve as better
chunk delimiters than sentence boundaries.
