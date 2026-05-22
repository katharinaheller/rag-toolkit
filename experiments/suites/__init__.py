"""Suite registry — every concrete suite exported here is picked up by the runner."""

from experiments.suites.base import ExperimentContext, Suite
from experiments.suites.s01_retriever_corpus_scaling import RetrieverCorpusScalingSuite
from experiments.suites.s02_topk_sensitivity import TopKSensitivitySuite
from experiments.suites.s03_query_type_comparison import QueryTypeSuite
from experiments.suites.s04_embedding_comparison import EmbeddingComparisonSuite
from experiments.suites.s05_latency_quality_pareto import LatencyQualityParetoSuite
from experiments.suites.s06_retrieval_overlap import RetrievalOverlapSuite
from experiments.suites.s07_hybrid_fusion_sweep import HybridFusionSweepSuite
from experiments.suites.s08_noise_robustness import NoiseRobustnessSuite
from experiments.suites.s09_stability import StabilitySuite
from experiments.suites.s10_context_pollution import ContextPollutionSuite
from experiments.suites.s11_long_context import LongContextSuite
from experiments.suites.s12_failure_taxonomy import FailureTaxonomySuite
from experiments.suites.s13_chunk_relevance_decay import ChunkRelevanceDecaySuite
from experiments.suites.s14_throughput_resources import ThroughputResourcesSuite
from experiments.suites.s15_query_conditioned import QueryConditionedSuite
from experiments.suites.s16_gpu_embedding_benchmark import GpuEmbeddingBenchmarkSuite
from experiments.suites.s17_gpu_generation_benchmark import GpuGenerationBenchmarkSuite
from experiments.suites.s18_batchsize_scaling import BatchsizeScalingSuite
from experiments.suites.s19_cold_warm import ColdWarmSuite
from experiments.suites.s20_resource_timeline import ResourceTimelineSuite
from experiments.suites.s21_concurrency_scaling import ConcurrencyScalingSuite


ALL_SUITES = [
    RetrieverCorpusScalingSuite,
    TopKSensitivitySuite,
    QueryTypeSuite,
    EmbeddingComparisonSuite,
    LatencyQualityParetoSuite,
    RetrievalOverlapSuite,
    HybridFusionSweepSuite,
    NoiseRobustnessSuite,
    StabilitySuite,
    ContextPollutionSuite,
    LongContextSuite,
    FailureTaxonomySuite,
    ChunkRelevanceDecaySuite,
    ThroughputResourcesSuite,
    QueryConditionedSuite,
    # GPU-aware benchmark + HPC resource suites.
    GpuEmbeddingBenchmarkSuite,
    GpuGenerationBenchmarkSuite,
    BatchsizeScalingSuite,
    ColdWarmSuite,
    ResourceTimelineSuite,
    ConcurrencyScalingSuite,
]


__all__ = [
    "ALL_SUITES",
    "ExperimentContext",
    "Suite",
]
