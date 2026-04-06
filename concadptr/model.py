"""
ConcAdptrModel — Core model class for ConcAdptr.

Composes a base transformer model with multiple LoRA adapters
and a learned routing network. This is the main user-facing class.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from peft import PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from concadptr.adapters import AdapterRegistry
from concadptr.config import ConcAdptrConfig, RoutingStrategy
from concadptr.router import BaseRouter, SoftMergingRouter, TopKRouter, XLoRARouter

logger = logging.getLogger(__name__)


class ConcAdptrModel(nn.Module):
    """ConcAdptr model: base model + multiple LoRA adapters + learned router.

    This is the main class for building a Mixture of LoRA Experts system.
    It handles loading the base model, attaching multiple LoRA adapters,
    initializing the routing network, and coordinating forward passes.

    The typical workflow is:
        1. Create from config: model = ConcAdptrModel.from_config(config)
        2. Train router: model.train_router(dataset)  # via ConcAdptrTrainer
        3. Inference: outputs = model.generate(inputs)
        4. Save: model.save_pretrained("./output")

    Args:
        config: ConcAdptrConfig with all settings.
    """

    def __init__(self, config: ConcAdptrConfig):
        super().__init__()
        self.config = config
        self.registry = AdapterRegistry()
        self.base_model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.router: Optional[BaseRouter] = None
        self._adapter_models: Dict[str, nn.Module] = {}
        self._is_loaded = False

    @classmethod
    def from_config(cls, config: ConcAdptrConfig) -> "ConcAdptrModel":
        """Create a ConcAdptrModel from a configuration.

        This is the primary constructor. It validates the config,
        loads the base model, attaches adapters, and initializes the router.

        Args:
            config: ConcAdptrConfig instance.

        Returns:
            Initialized ConcAdptrModel ready for router training or inference.
        """
        issues = config.validate()
        errors = [i for i in issues if i.startswith("ERROR")]
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(errors))
        for warning in [i for i in issues if i.startswith("WARNING")]:
            logger.warning(warning)

        model = cls(config)
        model._load_base_model()
        model._load_adapters()
        model._init_router()

        return model

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "ConcAdptrModel":
        """Create a ConcAdptrModel from a YAML configuration file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Initialized ConcAdptrModel.
        """
        config = ConcAdptrConfig.from_yaml(path)
        return cls.from_config(config)

    def _load_base_model(self) -> None:
        """Load the base transformer model with optional quantization."""
        logger.info(f"Loading base model: {self.config.base_model}")

        # Configure quantization
        quantization_config = None
        if self.config.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Resolve torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.bfloat16)

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Freeze base model
        if self.config.freeze_base_model:
            for param in self.base_model.parameters():
                param.requires_grad = False

        logger.info(f"Base model loaded: {self.base_model.__class__.__name__}")

    def _load_adapters(self) -> None:
        """Load and register all LoRA adapters from config."""
        logger.info(f"Loading {len(self.config.adapters)} adapters...")

        for name, path in self.config.adapters.items():
            # Register in the adapter registry
            try:
                self.registry.register(name, path)
            except FileNotFoundError:
                logger.warning(f"Adapter path not found: {path} ({name}), skipping")
                continue

        # Validate compatibility
        if self.registry.num_adapters >= 2:
            self.registry.validate_compatibility()

        # Load adapters into the model using PEFT
        first = True
        for adapter_info in self.registry:
            if first:
                # Load first adapter with PeftModel
                self.base_model = PeftModel.from_pretrained(
                    self.base_model,
                    adapter_info.path,
                    adapter_name=adapter_info.name,
                    is_trainable=not self.config.freeze_adapters,
                )
                first = False
            else:
                # Load additional adapters
                self.base_model.load_adapter(
                    adapter_info.path,
                    adapter_name=adapter_info.name,
                )

            adapter_info.loaded = True
            logger.info(f"  Loaded adapter: {adapter_info.name}")

        # Freeze adapters if configured
        if self.config.freeze_adapters:
            for name, param in self.base_model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = False

        self._is_loaded = True
        logger.info(f"All adapters loaded. Registry:\n{self.registry.summary()}")

    def _init_router(self) -> None:
        """Initialize the routing network based on config."""
        hidden_size = self.base_model.config.hidden_size
        num_layers = self.base_model.config.num_hidden_layers
        num_experts = self.registry.num_adapters

        strategy = self.config.router.strategy

        if strategy == RoutingStrategy.SOFT_MERGING:
            self.router = SoftMergingRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_layers=num_layers,
                use_layerwise=self.config.router.use_layerwise_routing,
                temperature=self.config.router.softmax_temperature,
                dropout=self.config.router.dropout,
            )
        elif strategy == RoutingStrategy.TOP_K:
            self.router = TopKRouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_layers=num_layers,
                use_layerwise=self.config.router.use_layerwise_routing,
                k=self.config.router.num_experts_per_token,
                temperature=self.config.router.softmax_temperature,
                dropout=self.config.router.dropout,
            )
        elif strategy == RoutingStrategy.XLORA:
            self.router = XLoRARouter(
                hidden_size=hidden_size,
                num_experts=num_experts,
                num_layers=num_layers,
                use_layerwise=self.config.router.use_layerwise_routing,
                classifier_depth=self.config.router.router_depth,
                classifier_hidden=self.config.router.router_hidden_size,
                temperature=self.config.router.softmax_temperature,
                dropout=self.config.router.dropout,
            )
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

        # Move router to same device as model
        device = next(self.base_model.parameters()).device
        self.router = self.router.to(device)

        # Count trainable params
        router_params = sum(p.numel() for p in self.router.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.base_model.parameters())
        logger.info(
            f"Router initialized: {strategy.value}, "
            f"{router_params:,} trainable params "
            f"({router_params / total_params * 100:.4f}% of model)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass with per-layer routed expert composition.

        Two-pass approach:
        1. Run base model with adapters disabled to capture per-layer hidden states.
        2. Compute per-layer routing weights from those hidden states, then run a
           single forward pass with hooks that apply weighted LoRA deltas at each layer.

        This replaces the old N-pass approach (one full pass per adapter) with a
        fixed 2-pass approach regardless of adapter count, and enables true layer-wise
        routing rather than routing only at the final logit level.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask.
            labels: Labels for computing language modeling loss.

        Returns:
            Dict with 'loss', 'lm_loss', 'logits', 'routing_weights',
            'routing_weights_per_layer', and 'load_balance_loss'.
        """
        adapter_names = self.registry.names
        # Avoid conflict if caller already set output_hidden_states
        kwargs.pop("output_hidden_states", None)

        # ── Pass 1: base model only (no LoRA) → per-layer hidden states ──────────
        with self.base_model.disable_adapter():
            base_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs,
            )

        # hidden_states[0] = embedding output; [1:] = after each transformer layer
        layer_hiddens = base_out.hidden_states[1:]

        # ── Compute per-layer routing weights ────────────────────────────────────
        layer_weights = []
        for layer_idx, hidden in enumerate(layer_hiddens):
            w = self.router(hidden.detach(), layer_idx=layer_idx)  # (batch, seq, num_experts)
            layer_weights.append(w)
        # (batch, seq, num_layers, num_experts)
        routing_weights_all = torch.stack(layer_weights, dim=2)

        # ── Pass 2: single forward with LoRA routing hooks ───────────────────────
        from peft.tuners.lora import LoraLayer

        hooks = []
        for module_name, module in self.base_model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            layer_idx = self._extract_layer_idx(module_name)
            hooks.append(
                module.register_forward_hook(
                    self._make_lora_hook(module, layer_idx, adapter_names, routing_weights_all)
                )
            )

        try:
            self.base_model.set_adapter(adapter_names[0])
            fwd_out = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        finally:
            for h in hooks:
                h.remove()

        fused_logits = fwd_out.logits

        # ── Losses ───────────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = fused_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # Mean routing across layers — gives every layer's gate gradient through lb_loss
        representative_weights = routing_weights_all.mean(dim=2)  # (batch, seq, num_experts)
        lb_loss = self.router.compute_load_balance_loss(representative_weights)
        total_loss = loss + self.config.router.load_balance_weight * lb_loss if loss is not None else None

        return {
            "loss": total_loss,
            "lm_loss": loss,
            "logits": fused_logits,
            "routing_weights": representative_weights.detach(),
            "routing_weights_per_layer": routing_weights_all.detach(),
            "load_balance_loss": lb_loss.detach(),
        }

    @staticmethod
    def _extract_layer_idx(module_name: str) -> int:
        """Parse the transformer layer index from a PEFT module name.

        Handles common patterns: .layers.N., .h.N., .blocks.N.
        Falls back to 0 if no match (e.g., embedding-level modules).
        """
        m = re.search(r"\.(layers|h|blocks)\.(\d+)\.", module_name)
        return int(m.group(2)) if m else 0

    @staticmethod
    def _make_lora_hook(
        lora_module,
        layer_idx: int,
        adapter_names: List[str],
        routing_weights_all: torch.Tensor,
    ):
        """Return a forward hook that replaces PEFT's single-adapter output with a
        routing-weighted combination of all adapters' LoRA deltas.

        Args:
            lora_module: The PEFT LoraLayer whose output will be overridden.
            layer_idx: Transformer layer index (indexes into routing_weights_all).
            adapter_names: Ordered list of adapter names.
            routing_weights_all: Pre-computed weights (batch, seq, num_layers, num_experts).
        """

        def hook(module, inp, output):
            x = inp[0]
            base_out = lora_module.base_layer(x)
            weights = routing_weights_all[:, :, layer_idx, :]  # (batch, seq, num_experts)

            weighted_delta = torch.zeros_like(base_out)
            for i, name in enumerate(adapter_names):
                if name not in lora_module.lora_A:
                    continue
                lora_dropout = (
                    lora_module.lora_dropout.get(name)
                    if hasattr(lora_module, "lora_dropout")
                    else None
                )
                x_in = lora_dropout(x) if lora_dropout is not None else x
                delta = lora_module.lora_B[name](lora_module.lora_A[name](x_in))
                delta = delta * lora_module.scaling[name]
                weighted_delta = weighted_delta + weights[..., i : i + 1] * delta

            return base_out + weighted_delta

        return hook

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get only the trainable parameters (router weights).

        Returns:
            List of trainable parameters for the optimizer.
        """
        return [p for p in self.router.parameters() if p.requires_grad]

    def get_num_trainable_params(self) -> int:
        """Get the number of trainable parameters."""
        return sum(p.numel() for p in self.get_trainable_parameters())

    def save_pretrained(self, path: Union[str, Path]) -> None:
        """Save the trained router and configuration.

        Only saves the router weights and fusion config (not the base model
        or adapters, which are referenced by path).

        Args:
            path: Directory to save the fusion model.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save router weights
        torch.save(self.router.state_dict(), path / "router.pt")

        # Save config
        self.config.save(path / "concadptr_config.yaml")

        # Save adapter registry info
        import json

        registry_info = {
            name: {
                "path": info.path,
                "rank": info.rank,
                "alpha": info.alpha,
                "target_modules": info.target_modules,
                "metadata": info.metadata,
            }
            for name, info in zip(self.registry.names, self.registry)
        }
        with open(path / "adapter_registry.json", "w") as f:
            json.dump(registry_info, f, indent=2)

        logger.info(f"ConcAdptr model saved to {path}")

    @classmethod
    def load_pretrained(cls, path: Union[str, Path]) -> "ConcAdptrModel":
        """Load a previously saved ConcAdptr model.

        Args:
            path: Directory containing the saved fusion model.

        Returns:
            Loaded ConcAdptrModel ready for inference.
        """
        path = Path(path)

        config = ConcAdptrConfig.from_yaml(path / "concadptr_config.yaml")
        model = cls.from_config(config)

        # Load router weights
        router_state = torch.load(path / "router.pt", map_location="cpu")
        model.router.load_state_dict(router_state)

        logger.info(f"ConcAdptr model loaded from {path}")
        return model

    def __repr__(self) -> str:
        parts = [
            f"ConcAdptrModel(",
            f"  base_model='{self.config.base_model}',",
            f"  num_adapters={self.registry.num_adapters},",
            f"  adapter_names={self.registry.names},",
            f"  routing={self.config.router.strategy.value},",
            f"  trainable_params={self.get_num_trainable_params():,}" if self.router else "",
            f")",
        ]
        return "\n".join(p for p in parts if p)
