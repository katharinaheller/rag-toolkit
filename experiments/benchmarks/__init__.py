"""GPU-aware benchmark runners for embedding, generation, scaling, concurrency.

Public surface::

    from experiments.benchmarks import (
        BenchmarkOutcome,
        EmbeddingBenchmark,
        GenerationBenchmark,
        ColdWarmBenchmark,
        ConcurrencyBenchmark,
    )

Every benchmark:

* always runs on CPU;
* runs on GPU only when ``torch.cuda.is_available()`` returns ``True``;
* never raises on a CPU-only host;
* records hardware metadata, raw values, computed summary, and resource
  timeline samples in a single :class:`BenchmarkOutcome`.
"""

from experiments.benchmarks.base import (
    BenchmarkOutcome,
    BenchmarkVariant,
    DeviceSpec,
    base_devices,
)
from experiments.benchmarks.embedding_benchmark import EmbeddingBenchmark
from experiments.benchmarks.generation_benchmark import GenerationBenchmark
from experiments.benchmarks.cold_warm_benchmark import ColdWarmBenchmark
from experiments.benchmarks.concurrency_benchmark import ConcurrencyBenchmark

__all__ = [
    "BenchmarkOutcome",
    "BenchmarkVariant",
    "ColdWarmBenchmark",
    "ConcurrencyBenchmark",
    "DeviceSpec",
    "EmbeddingBenchmark",
    "GenerationBenchmark",
    "base_devices",
]
