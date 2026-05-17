DOCUMENT_ID: doc_scale_050
TITLE: Benchmark Results for Hybrid Retrieval
CATEGORY: Hybrid Retrieval
CORPUS_SIZES: n50,n100
VARIANT: v10

CONTENT:
Hybrid retrieval with RRF consistently achieves state-of-the-art performance
on information retrieval benchmarks. On the MS MARCO Passage ranking
leaderboard, hybrid systems combining BM25 with dense bi-encoders achieve
nDCG@10 scores in the 0.75–0.82 range, outperforming BM25 alone (0.60–0.65)
and competitive with cross-encoder baselines. On the BEIR benchmark, hybrid
retrieval with BM25 + BGE-M3 achieves average nDCG@10 of approximately 0.59,
outperforming pure dense retrieval (0.53) and pure BM25 (0.49) across the
18-dataset evaluation. The gains are largest on domain-shifted datasets where
a dense model trained on MS MARCO generalises poorly, while BM25 provides a
reliable lexical fallback. These results support the practical recommendation
to default to hybrid retrieval with RRF as the first-stage retrieval strategy
in new RAG system deployments.
