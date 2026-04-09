"""
ConcAdptr Benchmarking Suite.

Provides task-specific metrics, MMLU/HellaSwag evaluation, and per-adapter
A/B comparison tooling.

Quick start::

    from concadptr import BenchmarkRunner, BenchmarkConfig

    runner = BenchmarkRunner(model, BenchmarkConfig(num_samples=200))

    # Evaluate full routing
    results = runner.run()

    # Compare base vs. each adapter vs. full routing
    comparison = runner.compare()

    # Detect catastrophic forgetting
    deltas = runner.forgetting_check()
"""

from concadptr.benchmarks.config import BenchmarkConfig, BenchmarkResult
from concadptr.benchmarks.metrics import accuracy, bleu, f1_score, rouge
from concadptr.benchmarks.runner import BenchmarkRunner

__all__ = [
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "accuracy",
    "f1_score",
    "bleu",
    "rouge",
]
