"""
ConcAdptr — Fuse multiple LoRA adapters into Mixture-of-Experts systems.

Train → Compose → Serve.

A library for composing independently trained LoRA adapters into
MoE-style expert systems with learned routing. Model-agnostic,
privacy-preserving, production-ready.

Quick Start:
    >>> from concadptr import ConcAdptrModel, ConcAdptrConfig
    >>> config = ConcAdptrConfig(
    ...     base_model="Qwen/Qwen2.5-7B-Instruct",
    ...     adapters={"medical": "./adapters/medical", "legal": "./adapters/legal"},
    ...     routing_strategy="xlora",
    ... )
    >>> model = ConcAdptrModel.from_config(config)
    >>> model.train_router(train_dataset)
    >>> model.save_pretrained("./fused_model")
"""

__version__ = "0.2.0"
__author__ = "Irfan Ali"

from concadptr.config import ConcAdptrConfig, MergeConfig, RouterConfig, ServingConfig
from concadptr.model import ConcAdptrModel
from concadptr.router import (
    BaseRouter,
    SoftMergingRouter,
    TopKRouter,
    XLoRARouter,
)
from concadptr.adapters import AdapterRegistry
from concadptr.merging import merge_adapters
from concadptr.trainer import ConcAdptrTrainer
from concadptr.benchmarks import BenchmarkConfig, BenchmarkResult, BenchmarkRunner

__all__ = [
    # Core
    "ConcAdptrModel",
    "ConcAdptrConfig",
    "RouterConfig",
    "ServingConfig",
    # Routers
    "BaseRouter",
    "SoftMergingRouter",
    "TopKRouter",
    "XLoRARouter",
    # Adapters
    "AdapterRegistry",
    # Merging
    "merge_adapters",
    "MergeConfig",
    # Training
    "ConcAdptrTrainer",
    # Benchmarking
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
]
