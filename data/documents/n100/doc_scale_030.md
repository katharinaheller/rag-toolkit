DOCUMENT_ID: doc_scale_030
TITLE: Batch Embedding for Large Corpora
CATEGORY: Dense Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v10

CONTENT:
Embedding a large document corpus with a dense encoder model involves
batched inference to amortise the overhead of model loading and tokenisation
across many inputs. The optimal batch size depends on available GPU memory,
sequence length distribution, and model architecture. For BGE-M3 with a
1024-dimensional output and 8192-token maximum sequence, a batch size of 16
to 64 is typical on a 16 GB GPU. Processing 100,000 passages typically takes
minutes on GPU hardware but hours on CPU, making GPU availability critical
for practical corpus-scale embedding. Checkpointing intermediate embeddings
to disk every N batches provides fault tolerance against OOM errors or
system interruptions. For very large corpora, distributed embedding across
multiple GPUs using model parallelism or data parallelism is standard in
production settings.
