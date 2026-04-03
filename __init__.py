"""
ConcAdpt — Fuse multiple LoRA adapters into Mixture-of-Experts systems.

Train → Compose → Serve.

A library for composing independently trained LoRA adapters into
MoE-style expert systems with learned routing. Model-agnostic,
privacy-preserving, production-ready.

Quick Start:
    >>> from concadpt import ConcAdptModel, ConcAdptConfig
    >>> config = ConcAdptConfig(
    ...     base_model="Qwen/Qwen2.5-7B-Instruct",
    ...     adapters={"medical": "./adapters/medical", "legal": "./adapters/legal"},
    ...     routing_strategy="xlora",
    ... )
    >>> model = ConcAdptModel.from_config(config)
    >>> model.train_router(train_dataset)
    >>> model.save_pretrained("./fused_model")
"""

__version__ = "0.1.0"
__author__ = "Irfan Ali"

from concadpt.config import ConcAdptConfig, RouterConfig, ServingConfig
from concadpt.model import ConcAdptModel
from concadpt.router import (
    BaseRouter,
    SoftMergingRouter,
    TopKRouter,
    XLoRARouter,
)
from concadpt.adapters import AdapterRegistry
from concadpt.trainer import ConcAdptTrainer

__all__ = [
    # Core
    "ConcAdptModel",
    "ConcAdptConfig",
    "RouterConfig",
    "ServingConfig",
    # Routers
    "BaseRouter",
    "SoftMergingRouter",
    "TopKRouter",
    "XLoRARouter",
    # Adapters
    "AdapterRegistry",
    # Training
    "ConcAdptTrainer",
]
