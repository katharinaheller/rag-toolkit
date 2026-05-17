DOCUMENT_ID: doc_scale_091
TITLE: Ollama: Local LLM Inference Server
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v1

CONTENT:
Ollama is an open-source local LLM inference server that provides a
REST API for running large language models on local hardware without
requiring cloud API access. Built on top of llama.cpp, Ollama handles model
downloading, quantisation, and serving through a simple command-line interface.
The server listens on port 11434 by default and exposes a REST API at
http://localhost:11434. Models are pulled using the 'ollama pull <model_name>'
command and stored locally in a model registry. The /api/generate endpoint
accepts POST requests with a JSON body containing the model name, prompt,
and generation parameters, and returns the generated text either as a
streaming response or a complete response depending on the 'stream' flag.
Ollama supports macOS (Metal GPU acceleration), Linux (CUDA, ROCm), and
Windows, making it accessible for development across all major platforms.
