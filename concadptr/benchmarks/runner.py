"""
BenchmarkRunner — orchestrates benchmark runs on a ConcAdptrModel.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from concadptr.benchmarks.config import BenchmarkConfig, BenchmarkResult
from concadptr.benchmarks.tasks import _TASK_REGISTRY, BenchmarkTask

if TYPE_CHECKING:
    from concadptr.model import ConcAdptrModel

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Orchestrates benchmark runs on a ConcAdptrModel.

    Supports three evaluation modes:

    - ``run()``: Evaluates all configured tasks using full ConcAdptr routing.
    - ``compare()``: Side-by-side comparison of base model, each individual adapter,
      and full routing. Useful for per-adapter A/B analysis.
    - ``forgetting_check()``: Runs MMLU + HellaSwag on both the base model and
      the routed model and reports accuracy deltas to detect catastrophic forgetting.

    Args:
        model: Initialized ConcAdptrModel.
        config: Benchmark configuration controlling which tasks to run and
            how many samples to evaluate.

    Example::

        runner = BenchmarkRunner(model, BenchmarkConfig(num_samples=100))
        results = runner.run()
        comparison = runner.compare()
        forgetting = runner.forgetting_check()
    """

    def __init__(self, model: ConcAdptrModel, config: BenchmarkConfig) -> None:
        self.model = model
        self.config = config

    def run(self, tasks: list[str] | None = None) -> list[BenchmarkResult]:
        """Run configured tasks on the full ConcAdptr routed model.

        Args:
            tasks: Task names to run. None falls back to config.tasks.

        Returns:
            List of BenchmarkResult, one per task.
        """
        task_names = tasks if tasks is not None else self.config.tasks
        results = self._run_tasks(
            task_names, adapter_name="concadptr_routed", collect_routing_stats=True
        )
        self._maybe_save(results)
        return results

    def compare(
        self,
        adapter_names: list[str] | None = None,
    ) -> dict[str, list[BenchmarkResult]]:
        """Compare base model, individual adapters, and full routing side by side.

        Runs all configured tasks under each model variant:
        - "base": all LoRA adapters disabled.
        - each adapter name: single-adapter mode (router bypassed).
        - "concadptr_routed": full ConcAdptr routing.

        Args:
            adapter_names: Adapter names to include in the comparison.
                None = all adapters registered in the model.

        Returns:
            Dict mapping variant name → list of BenchmarkResult. Keys are "base",
            each adapter name, and "concadptr_routed".
        """
        if adapter_names is None:
            adapter_names = self.model.registry.names

        variants: dict[str, list[BenchmarkResult]] = {}

        logger.info("Evaluating base model (no adapters)...")
        variants["base"] = self._run_tasks(self.config.tasks, adapter_name="base")

        for name in adapter_names:
            logger.info(f"Evaluating single adapter: {name}")
            variants[name] = self._run_tasks(self.config.tasks, adapter_name=name)

        logger.info("Evaluating concadptr_routed...")
        variants["concadptr_routed"] = self._run_tasks(
            self.config.tasks, adapter_name="concadptr_routed", collect_routing_stats=True
        )

        all_results = [r for rs in variants.values() for r in rs]
        self._maybe_save(all_results, filename="compare_results.json")
        return variants

    def forgetting_check(self) -> dict[str, dict[str, float]]:
        """Detect catastrophic forgetting relative to the base model.

        Runs MMLU and HellaSwag on both the base model (no adapters) and the full
        routed model, then reports accuracy deltas.

        Returns:
            Dict mapping task name → {"base": float, "routed": float, "delta": float}.
            A negative delta means the routed model lost accuracy on that benchmark.

        Example::

            {
                "mmlu":      {"base": 0.68, "routed": 0.67, "delta": -0.01},
                "hellaswag": {"base": 0.79, "routed": 0.78, "delta": -0.01},
            }
        """
        forgetting_tasks = ["mmlu", "hellaswag"]
        base_results = {
            r.task: r for r in self._run_tasks(forgetting_tasks, adapter_name="base")
        }
        routed_results = {
            r.task: r
            for r in self._run_tasks(
                forgetting_tasks,
                adapter_name="concadptr_routed",
                collect_routing_stats=True,
            )
        }

        comparison: dict[str, dict[str, float]] = {}
        for task_name in forgetting_tasks:
            base_acc = base_results[task_name].metrics.get("accuracy", 0.0)
            routed_acc = routed_results[task_name].metrics.get("accuracy", 0.0)
            comparison[task_name] = {
                "base": base_acc,
                "routed": routed_acc,
                "delta": round(routed_acc - base_acc, 6),
            }

        all_results = list(base_results.values()) + list(routed_results.values())
        self._maybe_save(all_results, filename="forgetting_check.json")
        return comparison

    def _run_tasks(
        self,
        task_names: list[str],
        adapter_name: str,
        collect_routing_stats: bool = False,
    ) -> list[BenchmarkResult]:
        """Run a list of tasks and return results.

        Args:
            task_names: Task names to run (must be registered in _TASK_REGISTRY).
            adapter_name: Model variant to pass to each task.
            collect_routing_stats: When True, enables router history before the run
                and attaches per-expert stats to every result afterward. Only
                meaningful when adapter_name == "concadptr_routed".

        Returns:
            List of BenchmarkResult in the same order as task_names.
        """
        results: list[BenchmarkResult] = []

        if collect_routing_stats:
            self.model.router.enable_history(True)

        try:
            for name in task_names:
                if name not in _TASK_REGISTRY:
                    logger.warning(f"Unknown benchmark task '{name}', skipping.")
                    continue
                task: BenchmarkTask = _TASK_REGISTRY[name]()
                logger.info(f"  Task '{name}' [{adapter_name}]...")
                result = task.evaluate(self.model, self.config, adapter_name=adapter_name)
                results.append(result)
        finally:
            if collect_routing_stats:
                self.model.router.enable_history(False)
                if results:
                    stats = self._collect_routing_stats()
                    for result in results:
                        result.routing_stats = stats

        return results

    def _collect_routing_stats(self) -> dict[str, float] | None:
        """Extract scalar routing statistics from router history.

        Converts tensor stats (expert_load, routing_entropy, expert_utilization)
        from router.get_routing_stats() to plain floats for JSON serialization.
        Non-scalar tensors (per-expert vectors) are averaged.

        Returns:
            Dict of scalar stat values, or None if no history was recorded.
        """
        stats = self.model.router.get_routing_stats()
        if not stats:
            return None
        out: dict[str, float] = {}
        for k, v in stats.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.item() if v.ndim == 0 else v.mean().item()
            else:
                out[k] = float(v)
        return out

    def _maybe_save(
        self,
        results: list[BenchmarkResult],
        filename: str = "benchmark_results.json",
    ) -> None:
        """Write results to JSON if config.output_dir is set."""
        if not self.config.output_dir:
            return
        path = Path(self.config.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "task": r.task,
                "adapter_name": r.adapter_name,
                "metrics": r.metrics,
                "num_samples": r.num_samples,
                "elapsed_seconds": r.elapsed_seconds,
                "routing_stats": r.routing_stats,
            }
            for r in results
        ]
        with open(path / filename, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Benchmark results saved to {path / filename}")
