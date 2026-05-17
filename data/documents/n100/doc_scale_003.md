DOCUMENT_ID: doc_scale_003
TITLE: RAG versus Parametric Language Models
CATEGORY: RAG Systems
CORPUS_SIZES: n10,n50,n100
VARIANT: v3

CONTENT:
Standard pre-trained language models store knowledge implicitly in their
weights during training. This parametric memory is fixed after training and
can become stale as the world changes. Retrieval-Augmented Generation
addresses this limitation by pairing the generative model with a dynamic,
updateable document store. Because the knowledge base can be refreshed
without retraining the model, RAG systems remain current at a fraction of the
cost of full model retraining. Furthermore, RAG provides natural attribution:
each generated statement can be traced back to a specific retrieved passage,
supporting auditability and user trust. The trade-off is increased inference
latency, since retrieval and prompt construction add overhead before the
generation step begins. Optimising this latency-quality frontier is an active
area of applied research in production RAG deployments.
