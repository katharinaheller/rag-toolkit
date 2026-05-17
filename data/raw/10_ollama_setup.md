# Ollama: Local LLM Serving

## Overview

Ollama is an open-source framework for running large language models locally on CPU or GPU hardware. It uses llama.cpp as its inference backend and provides a simple REST API for model management and text generation. Ollama handles model downloading, quantisation, and efficient inference without requiring the user to manage model weights directly.

Ollama is the primary generation backend for this RAG toolkit.

## Installation and Setup

Ollama runs as a background service and exposes a REST API. In Docker Compose environments, Ollama is typically configured as a separate service:

```yaml
services:
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
```

Ollama listens on port 11434 by default. The base URL for API calls is `http://ollama:11434` when accessed from other Docker containers, or `http://localhost:11434` when accessed from the host machine.

## API Endpoint

Ollama exposes a `/api/generate` endpoint that accepts a JSON request body and returns a JSON response:

```
POST http://ollama:11434/api/generate
```

Request body:
```json
{
  "model": "mistral:7b-instruct-q4_K_M",
  "prompt": "Your assembled prompt here",
  "stream": false,
  "options": {
    "temperature": 0.0,
    "num_predict": 512,
    "seed": 42,
    "top_p": 1.0,
    "repeat_penalty": 1.1
  }
}
```

Response body:
```json
{
  "model": "mistral:7b-instruct-q4_K_M",
  "response": "Generated answer text",
  "done": true,
  "total_duration": 12340000000
}
```

The `response` field contains the generated text. `stream: false` causes the full response to be returned in a single JSON object; this is the correct setting for synchronous evaluation workflows.

## Model Management

Pull a model before first use:
```
ollama pull mistral:7b-instruct-q4_K_M
```

List available models:
```
ollama list
```

## Reproducibility Configuration

For deterministic, reproducible generation:
- Set `temperature: 0.0` to disable sampling randomness.
- Set `seed: 42` to fix the random number generator seed.
- Set `stream: false` to receive the full response atomically.

Note: Reproducibility across different hardware, llama.cpp versions, or batch sizes is not guaranteed even with temperature=0 and a fixed seed, due to floating-point non-determinism in parallel reductions.

## Connection and Error Handling

Common failure modes when connecting to Ollama:

1. **Connection refused**: Ollama service is not running. Start it with `ollama serve`.
2. **Model not found**: The requested model has not been pulled. Run `ollama pull <model_name>`.
3. **Timeout**: Model is loading into RAM or generating a very long response. Increase the HTTP timeout.

A robust Ollama client should implement:
- Configurable HTTP timeout (default: 120 seconds).
- Fixed-delay retry on transient connection failures (default: 3 retries, 2-second delay).
- No retry on HTTP 4xx/5xx errors (these are deterministic and retrying is wasteful).

## Performance Notes

Inference speed depends critically on hardware:

- **CPU (4 cores, 8 GB RAM)**: 3–10 tokens/second with Mistral Q4_K_M.
- **GPU (RTX 3080, 10 GB VRAM)**: 20–100 tokens/second.

Cold start latency (first inference after loading) can be 8–15 seconds on CPU as the model weights are loaded from disk into RAM. Always perform at least one warmup call before benchmarking generation latency.

The `num_predict` parameter sets the maximum number of tokens to generate. Setting this too high (> 512) significantly increases evaluation time without improving answer quality for factoid questions.

## Docker Networking

In JupyterHub and Docker Compose setups, the Ollama service is accessed by its service name (not `localhost`). The default `base_url` in the GenerationConfig is `http://ollama:11434`. Override this with the environment variable `OLLAMA_BASE_URL` if your Ollama instance is accessible at a different address.
