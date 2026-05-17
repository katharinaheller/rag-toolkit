DOCUMENT_ID: doc_scale_097
TITLE: Environment Variable Configuration for RAG Systems
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v7

CONTENT:
Production RAG systems should externalise configuration through environment
variables to allow deployment without code changes. The OLLAMA_BASE_URL
variable controls the inference server endpoint and should default to
http://ollama:11434 in Docker environments and http://localhost:11434 for
local development. The EMBEDDING_DEVICE variable selects the compute device
for the embedding model (cpu, cuda, cuda:0). The INDEX_DIR variable specifies
the path to the persisted index, enabling different index versions to be
tested without code modifications. The RETRIEVAL_MODE variable selects the
retrieval strategy (dense, hybrid, sparse_bm25) at runtime. Using a
configuration management system or a .env file loaded at startup ensures that
all environment variables are set before the application initialises,
preventing hard-to-debug partial initialisation failures.
