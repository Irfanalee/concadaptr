"""
Soft Merging Router — MoLoRA-style routing.

Computes a weighted average across ALL experts for every token.
No sparsity — every expert contributes to every output.

Based on: Zadouri et al. (2023) "Pushing Mixture of Experts to the Limit"
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from concadpt.router.base import BaseRouter


class SoftMergingRouter(BaseRouter):
    """Dense/soft-merging router that blends all experts.

    Every token receives a weighted combination of all expert outputs.
    The weights are computed by a learned gating network.

    This is the simplest routing strategy and works well when:
    - The number of experts is small (2-8)
    - Domains overlap significantly
    - You want maximum knowledge sharing across experts

    Args:
        hidden_size: Hidden dimension of the base model.
        num_experts: Number of LoRA adapter experts.
        num_layers: Number of transformer layers.
        use_layerwise: Learn separate routing per layer.
        temperature: Softmax temperature (lower = sharper routing).
        dropout: Dropout probability in gating network.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_layers: int = 1,
        use_layerwise: bool = True,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_layers=num_layers,
            use_layerwise=use_layerwise,
        )

        self.temperature = temperature

        if use_layerwise:
            # Separate gate per layer
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
            # Shared gate across all layers
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
        """Compute soft routing weights for all experts.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            layer_idx: Current layer index (required if use_layerwise=True).

        Returns:
            Routing weights: (batch_size, seq_len, num_experts), sums to 1.
        """
        if self.use_layerwise:
            if layer_idx is None:
                raise ValueError("layer_idx required for layerwise routing")
            gate = self.gates[layer_idx]
        else:
            gate = self.gate

        # Compute logits and apply temperature-scaled softmax
        logits = gate(hidden_states)  # (batch, seq_len, num_experts)
        weights = F.softmax(logits / self.temperature, dim=-1)

        # Record history if enabled
        if self._record_history:
            self._routing_history.append(
                weights.detach().reshape(-1, self.num_experts).cpu()
            )

        return weights
