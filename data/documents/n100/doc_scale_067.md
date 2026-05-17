DOCUMENT_ID: doc_scale_067
TITLE: Domain-Specific Embedding Fine-Tuning
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v7

CONTENT:
General-purpose embedding models pre-trained on web-scale text may
underperform on domain-specific retrieval tasks due to distributional
mismatch. Fine-tuning on domain-specific query-passage pairs using a
contrastive objective can substantially improve retrieval quality, often
by 5-10 nDCG@10 points over the zero-shot baseline. The fine-tuning data
must include representative positive pairs (queries with relevant passages)
and hard negatives (queries with plausible but irrelevant passages from the
same domain). Synthetic data generation using a language model to write
hypothetical questions for each passage (the Hypothetical Document Embedding,
or HyDE approach) can bootstrap fine-tuning data when labelled pairs are
unavailable. Domain-specific fine-tuning is particularly impactful for
biomedical, legal, financial, and code retrieval domains where vocabulary
and linguistic style differ significantly from general web text.
