DOCUMENT_ID: doc_scale_093
TITLE: Docker Compose Integration for RAG Toolkits
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v3

CONTENT:
Running a RAG toolkit in a Docker Compose environment requires separate
services for the language model server, the retrieval API, and optionally
a web interface. The Ollama service uses the official ollama/ollama Docker
image and mounts a named volume to persist downloaded model weights across
container restarts. Other containers access Ollama at http://ollama:11434
using the Docker internal DNS name. The retrieval service should set the
OLLAMA_BASE_URL environment variable to override the default URL when running
inside Docker. GPU support requires setting the NVIDIA container runtime
and specifying device reservations in the Docker Compose service definition.
A health check on the Ollama container using 'ollama list' ensures that
dependent services do not start before the model server is ready.
