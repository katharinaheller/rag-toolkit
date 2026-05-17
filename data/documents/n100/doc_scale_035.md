DOCUMENT_ID: doc_scale_035
TITLE: FAISS Memory Management and GPU Support
CATEGORY: Vector Indexing
CORPUS_SIZES: n50,n100
VARIANT: v5

CONTENT:
FAISS provides native GPU support through the faiss-gpu package, which
implements GPU-accelerated flat and IVF index types. GPU indexes achieve
10 to 100 times higher throughput than CPU indexes for large batch queries,
making them essential for high-throughput retrieval services. Indexes can be
transferred between CPU and GPU memory using the index_cpu_to_gpu() function,
and multiple GPUs can be used in parallel through index_cpu_to_all_gpus().
For corpora that do not fit in GPU memory, sharding the index across multiple
GPUs and merging top-k results is the standard approach. The faiss-cpu package
provides optimised SIMD-accelerated implementations for Intel AVX2 and ARM
NEON, delivering significant speedups over generic C++ on modern server
hardware without requiring a GPU. Memory mapping (mmap) allows very large
flat indexes to be loaded from disk without loading the full index into RAM.
