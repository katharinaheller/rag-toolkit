DOCUMENT_ID: doc_scale_015
TITLE: BM25 Tokenisation and Preprocessing
CATEGORY: Sparse Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v5

CONTENT:
The quality of BM25 retrieval is highly sensitive to the tokenisation and
text normalisation pipeline applied before indexing. Standard preprocessing
steps include lowercasing to collapse case variants, stopword removal to
eliminate high-frequency function words that carry little semantic weight,
and stemming or lemmatisation to reduce inflected forms to their root. A
simple whitespace tokeniser treats punctuation as part of tokens and often
performs adequately for English technical text. More sophisticated tokenisers
apply compound word splitting, hyphen handling, and Unicode normalisation to
improve coverage. For domain-specific corpora such as biomedical or legal
text, specialised vocabularies and tokenisers tailored to that domain
consistently outperform generic approaches.
