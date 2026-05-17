# Mistral 7B: Architecture and Usage

## Overview

Mistral 7B is a large language model with 7 billion parameters released by Mistral AI in September 2023. Despite having fewer parameters than many competing models, Mistral 7B achieves state-of-the-art performance on multiple benchmarks, outperforming Llama 2 13B on all evaluated tasks and approaching Llama 2 34B performance in reasoning and coding benchmarks.

Mistral 7B uses several architectural innovations that improve inference efficiency and quality:

1. **Grouped Query Attention (GQA)**: Reduces the KV-cache size and inference latency by using fewer key/value heads than query heads.
2. **Sliding Window Attention (SWA)**: Allows the model to attend to sequences much longer than its training window through a sliding attention mechanism.
3. **Byte-Pair Encoding (BPE) Tokeniser**: Uses a vocabulary of 32,000 tokens with a SentencePiece tokeniser.

## Context Window

Mistral 7B supports a context window of 32,768 tokens (32k tokens) when using the sliding window attention mechanism. In practice, Ollama may serve a smaller effective context depending on the quantisation and hardware configuration.

## Instruction Fine-Tuning

Mistral 7B is available in both a base form and an instruction-tuned variant:

- **Mistral-7B-v0.1**: Base pre-trained model.
- **Mistral-7B-Instruct-v0.1**: Fine-tuned to follow instructions using a chat template format.
- **Mistral-7B-Instruct-v0.2**: Improved instruction following with better safety alignment.

For RAG applications, the Instruct variant is strongly preferred as it follows the system prompt and user instructions reliably.

## Quantisation

Quantisation reduces the precision of model weights to decrease memory consumption and inference latency at the cost of a small quality degradation. Common quantisation formats:

- **Q4_K_M**: 4-bit quantisation with mixed precision for key matrices. Approximately 4 GB RAM. Recommended balance of quality and efficiency.
- **Q5_K_M**: 5-bit quantisation. Approximately 5 GB RAM. Slightly better quality than Q4_K_M.
- **Q8_0**: 8-bit quantisation. Approximately 7.7 GB RAM. Near-original quality.

The Q4_K_M variant (`mistral:7b-instruct-q4_K_M`) is the standard choice for resource-constrained CPU inference in JupyterHub and HPC environments.

## Performance Characteristics

On a modern 4-core CPU with 8 GB RAM:

- **Cold start (model loading)**: 8–15 seconds
- **Per-token generation**: 10–30 milliseconds
- **Tokens per second**: 3–10 tokens/s depending on context length
- **512-token generation**: 50–170 seconds total

On a consumer GPU (e.g., RTX 3080):
- **Cold start**: 1–3 seconds
- **Per-token generation**: 1–5 milliseconds
- **Tokens per second**: 20–100 tokens/s

## Reproducibility

For reproducible RAG evaluation, always configure:
- `temperature: 0.0` — disables randomness in sampling
- `seed: 42` — fixes the random seed for deterministic outputs on the same hardware
- `repeat_penalty: 1.1` — mild penalty for repetition, improves output quality

Note: Perfect reproducibility is not guaranteed across different hardware, BLAS versions, or batch sizes, even with temperature=0 and a fixed seed.

## Prompt Format

Mistral Instruct models use a chat template with `[INST]` and `[/INST]` markers:

```
<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{user_message} [/INST]
```

For RAG applications, the system prompt should explicitly instruct the model to answer only from the provided context, preventing hallucination from parametric memory.

## Alternatives

Other models available via Ollama suitable for RAG:
- **Llama 3 8B Instruct**: Strong instruction following, 8 billion parameters.
- **Phi-3 Mini**: 3.8 billion parameters, strong reasoning, low memory footprint.
- **Gemma 2 9B**: Google's model, competitive with Mistral 7B in instruction following.
