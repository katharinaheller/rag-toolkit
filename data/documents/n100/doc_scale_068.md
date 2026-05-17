DOCUMENT_ID: doc_scale_068
TITLE: Multilingual Embedding Models
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v8

CONTENT:
Multilingual embedding models encode text from multiple languages into a
shared vector space, enabling cross-lingual retrieval where queries in one
language retrieve relevant passages in another. mE5 (multilingual E5) and
BGE-M3 are leading multilingual embedding models trained on data from 100+
languages. Cross-lingual retrieval performance degrades for low-resource
languages due to imbalanced training data, but models with sufficient
multilingual exposure achieve near-parity with monolingual models on the
MIRACL benchmark for major languages. For enterprise applications requiring
retrieval across a polyglot document corpus, a multilingual model eliminates
the need to maintain separate language-specific embedding models and indexes.
Language detection and routing to language-specific indexes is an alternative
architecture that provides better quality for high-traffic languages at the
cost of index fragmentation.
