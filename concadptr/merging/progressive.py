"""Progressive merging pipeline with quality gating.

Incrementally integrates new LoRA adapters into an existing concocted model,
running a forgetting check before each merge and rejecting adapters that
cause accuracy drops beyond the configured threshold.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from concadptr.benchmarks import BenchmarkConfig, BenchmarkRunner

if TYPE_CHECKING:
    from concadptr.model import ConcAdptrModel

logger = logging.getLogger(__name__)


class QualityGateError(Exception):
    """Raised when a candidate adapter fails the quality gate check.

    Args:
        message: Human-readable explanation of the failure.
        deltas: Per-task accuracy deltas that triggered the failure,
            e.g. ``{"mmlu": -0.05, "hellaswag": -0.01}``.
    """

    def __init__(self, message: str, deltas: dict[str, float]) -> None:
        super().__init__(message)
        self.deltas = deltas


@dataclass
class ProgressiveMergerConfig:
    """Configuration for the progressive merging pipeline.

    Args:
        quality_gate_threshold: Maximum allowed accuracy drop per task
            (negative means allowed drop). Adapters causing a delta below
            this value on any benchmark task are rejected.
            E.g. ``-0.02`` allows up to 2 % accuracy drop.
        merge_method: Static merging strategy — one of ``"linear"``,
            ``"ties"``, ``"dare"``, ``"dare_ties"``.
        merge_weights: Per-adapter blending coefficients for the merge.
            None = equal weights (computed automatically).
        ties_trim_fraction: TIES only — fraction of low-magnitude
            parameters to zero out before merging.
        dare_density: DARE only — fraction of parameters to keep (drop rate
            = 1 - density).
        benchmark_tasks: Tasks to run for the forgetting check.
        benchmark_num_samples: Number of examples per task. Keep small
            (e.g. 50) for fast gating; None = full dataset.
    """

    quality_gate_threshold: float = -0.02
    merge_method: str = "linear"
    merge_weights: list[float] | None = None
    ties_trim_fraction: float = 0.2
    dare_density: float = 0.7
    benchmark_tasks: list[str] = field(default_factory=lambda: ["mmlu", "hellaswag"])
    benchmark_num_samples: int | None = 50


@dataclass
class MergeResult:
    """Result returned by ProgressiveMerger.add_adapter().

    Args:
        adapter_name: Name of the adapter that was evaluated.
        output_path: Directory path where the merged adapter was written.
        deltas: Per-task accuracy deltas from the forgetting check,
            e.g. ``{"mmlu": -0.01, "hellaswag": 0.00}``.
        passed_gate: Whether the adapter passed the quality gate.
        merged: Whether the merge was performed. False when ``dry_run=True``.
    """

    adapter_name: str
    output_path: Path
    deltas: dict[str, float]
    passed_gate: bool
    merged: bool


class ProgressiveMerger:
    """Incremental adapter integration with quality gating.

    When a new adapter arrives, registers it, runs a forgetting check against
    the base model, and calls ``merge_adapters()`` only if accuracy deltas on
    all benchmark tasks stay within the configured threshold.

    Args:
        model: Initialized ConcAdptrModel with existing adapters registered.
        config: Gating and merging configuration.

    Example::

        merger = ProgressiveMerger(model, ProgressiveMergerConfig(
            quality_gate_threshold=-0.02,
            merge_method="ties",
        ))
        try:
            result = merger.add_adapter(
                "finance", "./adapters/finance", "./merged_v2"
            )
            print(f"Merged to {result.output_path}, deltas={result.deltas}")
        except QualityGateError as e:
            print(f"Adapter rejected. Deltas: {e.deltas}")
    """

    def __init__(
        self,
        model: ConcAdptrModel,
        config: ProgressiveMergerConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or ProgressiveMergerConfig()

    def add_adapter(
        self,
        name: str,
        path: str | Path,
        output_path: str | Path,
        *,
        dry_run: bool = False,
    ) -> MergeResult:
        """Register a new adapter, gate on quality, and merge if it passes.

        Steps:
        1. Register the adapter in the model registry (validates PEFT format).
        2. Run ``BenchmarkRunner.forgetting_check()`` — base vs routed accuracy.
        3. If any task delta < threshold → unregister adapter and raise
           ``QualityGateError``.
        4. If ``dry_run=True`` → return result without merging.
        5. Merge all registered adapters to ``output_path``.

        Args:
            name: Unique name for the adapter.
            path: Path to the PEFT adapter directory.
            output_path: Directory to write the merged adapter.
            dry_run: If True, run the gate check but skip the merge step.

        Returns:
            MergeResult with per-task deltas, gate status, and merge status.

        Raises:
            QualityGateError: If any task accuracy drop exceeds the threshold.
            FileNotFoundError: If ``path`` does not exist (from registry).
            ValueError: If ``name`` is already registered (from registry).
        """
        path = Path(path)
        output_path = Path(output_path)

        # Step 1: register adapter (validates PEFT format)
        self.model.registry.register(name, path)
        logger.info(f"Registered adapter '{name}', running quality gate...")

        # Step 2: forgetting check — base vs routed accuracy
        bench_config = BenchmarkConfig(
            tasks=self.config.benchmark_tasks,
            num_samples=self.config.benchmark_num_samples,
        )
        runner = BenchmarkRunner(self.model, bench_config)
        forgetting = runner.forgetting_check()

        # Step 3: gate check — fail if ANY task drops below threshold
        deltas = {task: result["delta"] for task, result in forgetting.items()}
        failed = {
            task: delta
            for task, delta in deltas.items()
            if delta < self.config.quality_gate_threshold
        }

        if failed:
            self.model.registry.unregister(name)
            worst = min(failed, key=lambda t: failed[t])
            raise QualityGateError(
                f"Adapter '{name}' failed quality gate: "
                f"{worst} delta={failed[worst]:.4f} "
                f"< threshold={self.config.quality_gate_threshold}",
                deltas=deltas,
            )

        logger.info(f"Adapter '{name}' passed quality gate. Deltas: {deltas}")

        # Step 4: dry run — skip merge
        if dry_run:
            return MergeResult(
                adapter_name=name,
                output_path=output_path,
                deltas=deltas,
                passed_gate=True,
                merged=False,
            )

        # Step 5: merge all registered adapters (including the new one)
        from concadptr.merging import merge_adapters

        adapter_paths = {info.name: info.path for info in self.model.registry}
        weights = self._compute_weights(len(adapter_paths))

        merge_adapters(
            adapters=adapter_paths,
            output_path=output_path,
            method=self.config.merge_method,
            weights=weights,
            trim_fraction=self.config.ties_trim_fraction,
            density=self.config.dare_density,
        )

        logger.info(f"Merged {len(adapter_paths)} adapters → {output_path}")

        return MergeResult(
            adapter_name=name,
            output_path=output_path,
            deltas=deltas,
            passed_gate=True,
            merged=True,
        )

    def _compute_weights(self, n: int) -> list[float]:
        """Return merge weights: config weights if set, else equal weights."""
        if self.config.merge_weights is not None:
            return self.config.merge_weights
        return [1.0 / n] * n
