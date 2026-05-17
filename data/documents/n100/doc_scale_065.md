DOCUMENT_ID: doc_scale_065
TITLE: Embedding Model Caching and Inference Optimisation
CATEGORY: Embedding Models
CORPUS_SIZES: n100
VARIANT: v5

CONTENT:
Loading an embedding model from disk and initialising its weights takes several
seconds, making it important to load the model once and reuse it across all
embedding calls rather than reloading for each batch. A model cache stores
loaded model instances keyed by model name, device, and precision settings,
allowing multiple pipeline components to share the same model instance.
Inference optimisation techniques include dynamic batching (grouping short and
long sequences together to minimise padding waste), TorchScript compilation,
ONNX export with ONNX Runtime inference, and 8-bit quantisation for CPU
deployment. Flash Attention 2 significantly reduces memory and latency for
long-sequence encoding. For production serving, Triton Inference Server or
vLLM with embedding mode support can handle high request throughput with
GPU batching and kernel fusion.
