"""
Configuration dataclasses for ConcAdptr benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmarking run.

    Args:
        tasks: Task names to run. Recognized values: "mmlu", "hellaswag", "generation".
            Defaults to ["mmlu", "hellaswag"].
        num_samples: Maximum examples to evaluate per task. None = full dataset.
        batch_size: Reserved for future batched inference; currently examples are
            processed one at a time for simplicity.
        max_new_tokens: Number of tokens to generate per example (GenerationTask only).
        mmlu_subjects: Subset of MMLU subjects to evaluate. None = all subjects.
            Example: ["abstract_algebra", "anatomy"].
        mmlu_split: HuggingFace dataset split for MMLU (typically "test").
        hellaswag_split: HuggingFace dataset split for HellaSwag (typically "validation").
        generation_dataset: HuggingFace dataset name for GenerationTask.
            Required when "generation" is in tasks.
        generation_input_field: Column name used as model input in the generation dataset.
        generation_reference_field: Column name used as reference output.
        generation_split: Dataset split for the generation dataset.
        generation_metrics: Metrics to compute for generation. Supported: "bleu", "rouge".
            Requires pip install 'concadptr[benchmarks]'.
        output_dir: Optional directory to write JSON result files.
    """

    tasks: list[str] = field(default_factory=lambda: ["mmlu", "hellaswag"])
    num_samples: int | None = None
    batch_size: int = 4
    max_new_tokens: int = 64
    mmlu_subjects: list[str] | None = None
    mmlu_split: str = "test"
    hellaswag_split: str = "validation"
    generation_dataset: str | None = None
    generation_input_field: str = "input"
    generation_reference_field: str = "output"
    generation_split: str = "test"
    generation_metrics: list[str] = field(default_factory=lambda: ["bleu", "rouge"])
    output_dir: str | None = None


@dataclass
class BenchmarkResult:
    """Result from evaluating one task on one model variant.

    Args:
        task: Name of the benchmark task (e.g. "mmlu", "hellaswag", "generation").
        adapter_name: Model variant evaluated. "base" = no adapters, any other string
            selects that named adapter, "concadptr_routed" = full routing.
        metrics: Metric name → value. E.g. {"accuracy": 0.72} or {"bleu": 34.1}.
        num_samples: Number of examples evaluated.
        elapsed_seconds: Wall-clock time for the run.
        routing_stats: Per-expert load and entropy statistics, populated only when
            adapter_name == "concadptr_routed" and routing history was collected.
    """

    task: str
    adapter_name: str
    metrics: dict[str, float]
    num_samples: int
    elapsed_seconds: float
    routing_stats: dict[str, float] | None = None
