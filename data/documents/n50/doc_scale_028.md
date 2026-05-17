DOCUMENT_ID: doc_scale_028
TITLE: Zero-Shot Dense Retrieval
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v8

CONTENT:
Zero-shot dense retrieval refers to applying a pre-trained dense retrieval
model to a target domain without any domain-specific fine-tuning. Models
trained on general-domain corpora such as MS MARCO sometimes generalise well
to new domains, but domain shift remains a significant challenge. BEIR
(Benchmarking IR) is a heterogeneous benchmark containing 18 retrieval
datasets spanning scientific papers, financial news, medical documents, and
code that is specifically designed to evaluate zero-shot generalisation.
Results on BEIR show that dense models trained on MS MARCO underperform
BM25 on several out-of-domain datasets, while multilingual models like
BGE-M3 that are trained on more diverse data generalise substantially better.
Instruction-tuned retrieval models, which accept a natural language task
description alongside the query, represent a promising direction for
improving zero-shot cross-domain retrieval.
