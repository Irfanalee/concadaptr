"""
Configuration classes for ConcAdpt.

Provides structured configs for model composition, router training, and serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


class RoutingStrategy(str, Enum):
    """Available routing strategies for adapter composition."""

    SOFT_MERGING = "soft_merging"  # Weighted average of all experts (MoLoRA-style)
    TOP_K = "top_k"  # Activate top-k experts per token (MixLoRA-style)
    XLORA = "xlora"  # Learned scaling with frozen adapters (X-LoRA-style)


class MergeMethod(str, Enum):
    """Static merge methods (non-routing alternatives)."""

    LINEAR = "linear"
    TIES = "ties"
    DARE = "dare"
    DARE_TIES = "dare_ties"


@dataclass
class RouterConfig:
    """Configuration for the routing/gating network.

    Args:
        strategy: Which routing approach to use.
        num_experts_per_token: Number of experts activated per token (for top_k).
        router_hidden_size: Hidden dimension of the router MLP.
        router_depth: Number of layers in the router MLP.
        softmax_temperature: Temperature for softmax in routing distribution.
        load_balance_weight: Weight of the auxiliary load-balancing loss.
        dropout: Dropout probability in the router.
        use_layerwise_routing: Whether to learn separate routing per transformer layer.
    """

    strategy: RoutingStrategy = RoutingStrategy.XLORA
    num_experts_per_token: int = 2
    router_hidden_size: int = 256
    router_depth: int = 1
    softmax_temperature: float = 1.0
    load_balance_weight: float = 0.01
    dropout: float = 0.1
    use_layerwise_routing: bool = True


@dataclass
class TrainingConfig:
    """Configuration for router training.

    Args:
        learning_rate: Learning rate for router parameters.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        max_seq_length: Maximum sequence length for training examples.
        warmup_ratio: Proportion of training for learning rate warmup.
        weight_decay: Weight decay coefficient.
        gradient_accumulation_steps: Number of gradient accumulation steps.
        fp16: Use mixed precision training.
        logging_steps: Log metrics every N steps.
        eval_steps: Run evaluation every N steps.
        save_steps: Save checkpoint every N steps.
        output_dir: Directory for saving checkpoints and logs.
    """

    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    output_dir: str = "./concadpt_output"


@dataclass
class ServingConfig:
    """Configuration for inference serving.

    Args:
        host: Server host address.
        port: Server port.
        max_concurrent_requests: Maximum concurrent inference requests.
        adapter_cache_size: Number of adapters to keep in GPU memory.
        default_adapter: Default adapter for requests without explicit routing.
        enable_metrics: Enable Prometheus-compatible metrics endpoint.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    max_concurrent_requests: int = 32
    adapter_cache_size: int = 10
    default_adapter: Optional[str] = None
    enable_metrics: bool = True


@dataclass
class ConcAdptConfig:
    """Top-level configuration for ConcAdpt.

    This is the main configuration class that ties everything together.
    It can be created programmatically or loaded from a YAML file.

    Args:
        base_model: HuggingFace model ID or local path to the base model.
        adapters: Dictionary mapping adapter names to their paths.
            Example: {"medical": "./adapters/medical", "legal": "./adapters/legal"}
        routing_strategy: Shorthand for setting router.strategy.
        router: Detailed router configuration.
        training: Training configuration for the router.
        serving: Serving configuration for inference.
        quantization: Quantization mode for base model ("4bit", "8bit", or None).
        torch_dtype: Data type for model weights ("float16", "bfloat16", "float32").
        device_map: Device mapping strategy ("auto", "cuda:0", etc.).
        trust_remote_code: Whether to trust remote code in HF models.
        freeze_adapters: Whether to freeze adapter weights during router training.
        freeze_base_model: Whether to freeze base model weights (should always be True).

    Example:
        >>> config = ConcAdptConfig(
        ...     base_model="Qwen/Qwen2.5-7B-Instruct",
        ...     adapters={
        ...         "medical": "./adapters/medical",
        ...         "legal": "./adapters/legal",
        ...         "finance": "./adapters/finance",
        ...     },
        ...     routing_strategy="xlora",
        ... )
    """

    base_model: str = ""
    adapters: Dict[str, str] = field(default_factory=dict)
    routing_strategy: str = "xlora"

    router: RouterConfig = field(default_factory=RouterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    serving: ServingConfig = field(default_factory=ServingConfig)

    quantization: Optional[str] = "4bit"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    freeze_adapters: bool = True
    freeze_base_model: bool = True

    def __post_init__(self):
        """Sync routing_strategy shorthand with router config."""
        if self.routing_strategy:
            self.router.strategy = RoutingStrategy(self.routing_strategy)

    def validate(self) -> List[str]:
        """Validate the configuration and return a list of warnings/errors.

        Returns:
            List of validation messages. Empty list means valid.
        """
        issues = []

        if not self.base_model:
            issues.append("ERROR: base_model is required")

        if not self.adapters:
            issues.append("ERROR: at least one adapter must be specified")

        if len(self.adapters) < 2:
            issues.append(
                "WARNING: ConcAdpt works best with 2+ adapters. "
                "With a single adapter, consider using PEFT directly."
            )

        for name, path in self.adapters.items():
            adapter_path = Path(path)
            if not adapter_path.exists():
                issues.append(f"WARNING: adapter path does not exist: {path} ({name})")
            else:
                # Check for adapter_config.json or adapter_model.safetensors
                has_config = (adapter_path / "adapter_config.json").exists()
                has_weights = (
                    (adapter_path / "adapter_model.safetensors").exists()
                    or (adapter_path / "adapter_model.bin").exists()
                )
                if not has_config:
                    issues.append(
                        f"WARNING: missing adapter_config.json in {path} ({name})"
                    )
                if not has_weights:
                    issues.append(
                        f"WARNING: missing adapter weights in {path} ({name})"
                    )

        if self.router.strategy == RoutingStrategy.TOP_K:
            if self.router.num_experts_per_token > len(self.adapters):
                issues.append(
                    f"WARNING: num_experts_per_token ({self.router.num_experts_per_token}) "
                    f"> number of adapters ({len(self.adapters)}). "
                    f"This is equivalent to soft merging."
                )

        if self.quantization not in (None, "4bit", "8bit"):
            issues.append(
                f"ERROR: quantization must be None, '4bit', or '8bit', got '{self.quantization}'"
            )

        return issues

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            path: File path to save the configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "base_model": self.base_model,
            "adapters": self.adapters,
            "routing_strategy": self.routing_strategy,
            "quantization": self.quantization,
            "torch_dtype": self.torch_dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "freeze_adapters": self.freeze_adapters,
            "freeze_base_model": self.freeze_base_model,
            "router": {
                "strategy": self.router.strategy.value,
                "num_experts_per_token": self.router.num_experts_per_token,
                "router_hidden_size": self.router.hidden_size
                if hasattr(self.router, "hidden_size")
                else self.router.router_hidden_size,
                "router_depth": self.router.router_depth,
                "softmax_temperature": self.router.softmax_temperature,
                "load_balance_weight": self.router.load_balance_weight,
                "dropout": self.router.dropout,
                "use_layerwise_routing": self.router.use_layerwise_routing,
            },
            "training": {
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "batch_size": self.training.batch_size,
                "max_seq_length": self.training.max_seq_length,
                "warmup_ratio": self.training.warmup_ratio,
                "weight_decay": self.training.weight_decay,
                "output_dir": self.training.output_dir,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ConcAdptConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            ConcAdptConfig instance.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        router_data = data.pop("router", {})
        training_data = data.pop("training", {})
        serving_data = data.pop("serving", {})

        router_config = RouterConfig(
            strategy=RoutingStrategy(router_data.get("strategy", "xlora")),
            num_experts_per_token=router_data.get("num_experts_per_token", 2),
            router_hidden_size=router_data.get("router_hidden_size", 256),
            router_depth=router_data.get("router_depth", 1),
            softmax_temperature=router_data.get("softmax_temperature", 1.0),
            load_balance_weight=router_data.get("load_balance_weight", 0.01),
            dropout=router_data.get("dropout", 0.1),
            use_layerwise_routing=router_data.get("use_layerwise_routing", True),
        )

        training_config = TrainingConfig(**{
            k: v for k, v in training_data.items() if k in TrainingConfig.__dataclass_fields__
        }) if training_data else TrainingConfig()

        serving_config = ServingConfig(**{
            k: v for k, v in serving_data.items() if k in ServingConfig.__dataclass_fields__
        }) if serving_data else ServingConfig()

        return cls(
            router=router_config,
            training=training_config,
            serving=serving_config,
            **{k: v for k, v in data.items() if k in cls.__dataclass_fields__},
        )
