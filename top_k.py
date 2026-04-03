"""
Top-K Router — MixLoRA-style sparse routing.

Only the top-k experts are activated per token, providing sparsity
and computational efficiency. Inactive experts contribute zero output.

Based on: Li et al. (2024) "MixLoRA: Enhancing LLMs Fine-Tuning with LoRA-based MoE"
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from concadpt.router.base import BaseRouter


class TopKRouter(BaseRouter):
    """Sparse top-k router that activates only k experts per token.

    For each token, the router computes scores for all experts, then
    selects the top-k experts and renormalizes their weights. Non-selected
    experts receive zero weight.

    This is efficient when:
    - You have many experts (8+)
    - Experts are highly specialized (low domain overlap)
    - You want to control compute cost per token

    Args:
        hidden_size: Hidden dimension of the base model.
        num_experts: Number of LoRA adapter experts.
        num_layers: Number of transformer layers.
        use_layerwise: Learn separate routing per layer.
        k: Number of experts to activate per token.
        temperature: Softmax temperature.
        dropout: Dropout in gating network.
        add_noise: Add noise to routing logits during training (helps exploration).
        noise_std: Standard deviation of routing noise.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_layers: int = 1,
        use_layerwise: bool = True,
        k: int = 2,
        temperature: float = 1.0,
        dropout: float = 0.1,
        add_noise: bool = True,
        noise_std: float = 0.1,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_layers=num_layers,
            use_layerwise=use_layerwise,
        )

        self.k = min(k, num_experts)
        self.temperature = temperature
        self.add_noise = add_noise
        self.noise_std = noise_std

        if use_layerwise:
            self.gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 4, num_experts),
                )
                for _ in range(num_layers)
            ])
        else:
            self.gate = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 4, num_experts),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute sparse top-k routing weights.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            layer_idx: Current layer index.

        Returns:
            Routing weights: (batch_size, seq_len, num_experts).
            Only top-k entries are non-zero per token.
        """
        if self.use_layerwise:
            if layer_idx is None:
                raise ValueError("layer_idx required for layerwise routing")
            gate = self.gates[layer_idx]
        else:
            gate = self.gate

        logits = gate(hidden_states)  # (batch, seq_len, num_experts)

        # Add noise during training for exploration
        if self.training and self.add_noise:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Get top-k indices and values
        top_k_logits, top_k_indices = logits.topk(self.k, dim=-1)  # (batch, seq, k)

        # Create sparse weight tensor
        # Softmax only over the selected experts
        top_k_weights = F.softmax(top_k_logits / self.temperature, dim=-1)  # (batch, seq, k)

        # Scatter back to full expert dimension
        weights = torch.zeros_like(logits)  # (batch, seq, num_experts)
        weights.scatter_(-1, top_k_indices, top_k_weights)

        # Record history if enabled
        if self._record_history:
            self._routing_history.append(
                weights.detach().reshape(-1, self.num_experts).cpu()
            )

        return weights
