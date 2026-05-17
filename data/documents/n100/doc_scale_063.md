DOCUMENT_ID: doc_scale_063
TITLE: Sentence-BERT and S-BERT Variants
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v3

CONTENT:
Sentence-BERT (S-BERT) introduced the technique of fine-tuning BERT with
siamese and triplet network structures using contrastive objectives to produce
semantically meaningful sentence embeddings. Before S-BERT, BERT's output
required the full cross-encoder pass for each pair, making large-scale semantic
similarity computation intractable. S-BERT decouples query and document
encoding, enabling pre-computation of document embeddings and O(1) similarity
scoring via dot product. The sentence-transformers library provides hundreds
of pre-trained S-BERT variants optimised for different languages, domains,
and embedding dimensions. Models in the all-mpnet-base-v2 and all-MiniLM-L6-v2
families remain popular baselines for retrieval tasks, balancing quality and
inference speed. Larger models such as multi-qa-mpnet-base-dot-v1 are
specifically optimised for question-passage matching retrieval tasks.
