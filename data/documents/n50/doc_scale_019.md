DOCUMENT_ID: doc_scale_019
TITLE: BM25 for Code Retrieval
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v9

CONTENT:
Applying BM25 to code retrieval requires adaptations to handle the distinct
token distribution of programming languages. Code tokens such as function
names, variable names, and API identifiers are often highly unique, making
BM25's IDF weighting particularly effective at distinguishing relevant code
snippets from generic boilerplate. However, programming language syntax
introduces tokens like brackets, semicolons, and operators that function as
noise in a lexical retrieval context. Best practice for code retrieval with
BM25 involves separating identifier tokens from syntax tokens, lowercasing
camel-case identifiers by splitting on case boundaries, and applying a
tokeniser that understands subword units common in code identifiers.
BM25 code retrieval forms the baseline in code search benchmarks such as
CodeSearchNet, against which dense code embedding models are evaluated.
