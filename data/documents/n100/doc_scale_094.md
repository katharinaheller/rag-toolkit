DOCUMENT_ID: doc_scale_094
TITLE: GPU vs CPU Inference with Ollama
CATEGORY: Infrastructure
CORPUS_SIZES: n100
VARIANT: v4

CONTENT:
Ollama automatically detects available GPU hardware and offloads model layers
to the GPU when sufficient VRAM is available. For Mistral 7B in Q4_K_M
quantisation (approximately 4.1 GB), a 6 GB VRAM GPU can run the full model
on GPU, achieving 20-50 tokens per second. When VRAM is insufficient, Ollama
partially offloads the model, placing some layers on GPU and the remaining
layers on CPU, with the split configurable through the OLLAMA_GPU_LAYERS
environment variable. CPU-only inference is fully supported and produces
identical outputs to GPU inference due to the deterministic nature of greedy
decoding. For RAG evaluation workloads requiring deterministic, reproducible
results, CPU inference eliminates variability introduced by GPU floating-point
non-determinism at the cost of higher latency.
