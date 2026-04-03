"""
X-LoRA Router — Learned scaling with frozen adapters.

The X-LoRA approach freezes all adapter weights and learns only the
scaling/routing coefficients. This is the most practical approach when
adapters are trained independently (e.g., at different customer sites).

Based on: Buehler & Buehler (2024) "X-LoRA: Mixture of Low-Rank Adapter Experts"
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from concadpt.router.base import BaseRouter


class XLoRARouter(BaseRouter):
    """X-LoRA router: learns scaling values for frozen LoRA adapters.

    This router is specifically designed for the case where LoRA adapters
    are trained independently and then composed. The key properties are:

    1. All adapters remain FROZEN — only the router is trained
    2. Dense gating — all adapters contribute (weighted) to every output
    3. Layer-wise scaling — each transformer layer gets its own routing
    4. Low parameter count — efficient to train on small datasets

    This is the recommended router for the multi-customer architecture
    because adapters from different customer environments can be plugged
    in without retraining them.

    Args:
        hidden_size: Hidden dimension of the base model.
        num_experts: Number of LoRA adapter experts.
        num_layers: Number of transformer layers.
        use_layerwise: Learn separate scaling per layer (strongly recommended).
        classifier_depth: Depth of the classifier MLP.
        classifier_hidden: Hidden size of the classifier MLP.
        temperature: Softmax temperature.
        dropout: Dropout probability.
        enable_softmax: Apply softmax normalization to scaling values.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_layers: int = 1,
        use_layerwise: bool = True,
        classifier_depth: int = 1,
        classifier_hidden: int = 2048,
        temperature: float = 1.0,
        dropout: float = 0.2,
        enable_softmax: bool = True,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_experts=num_experts,
            num_layers=num_layers,
            use_layerwise=use_layerwise,
        )

        self.temperature = temperature
        self.enable_softmax = enable_softmax

        # Number of scaling outputs:
        # If layerwise: one set of expert weights per layer
        # If not: single set of expert weights
        output_size = num_experts * num_layers if use_layerwise else num_experts

        # Build classifier MLP
        layers = []
        in_size = hidden_size

        for i in range(classifier_depth):
            out_size = classifier_hidden if i < classifier_depth - 1 else output_size
            layers.append(nn.Linear(in_size, out_size))
            if i < classifier_depth - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
            in_size = out_size

        self.classifier = nn.Sequential(*layers)

        # Scaling bias — initialized to uniform distribution
        self.scaling_bias = nn.Parameter(
            torch.zeros(num_layers if use_layerwise else 1, num_experts)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute X-LoRA scaling values for adapter composition.

        The classifier processes the hidden states (typically from the
        last layer or a pooled representation) and produces per-expert
        scaling values.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            layer_idx: Current layer index. If provided with layerwise routing,
                returns scaling for this specific layer.

        Returns:
            Scaling weights: (batch_size, seq_len, num_experts).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute raw scaling logits
        logits = self.classifier(hidden_states)  # (batch, seq, output_size)

        if self.use_layerwise:
            # Reshape to (batch, seq, num_layers, num_experts)
            logits = logits.reshape(batch_size, seq_len, self.num_layers, self.num_experts)

            # Add bias
            logits = logits + self.scaling_bias.unsqueeze(0).unsqueeze(0)

            if layer_idx is not None:
                # Return scaling for specific layer
                layer_logits = logits[:, :, layer_idx, :]  # (batch, seq, num_experts)
            else:
                # Return average scaling across all layers
                layer_logits = logits.mean(dim=2)  # (batch, seq, num_experts)
        else:
            layer_logits = logits + self.scaling_bias.squeeze(0).unsqueeze(0).unsqueeze(0)

        # Apply softmax for normalization
        if self.enable_softmax:
            weights = F.softmax(layer_logits / self.temperature, dim=-1)
        else:
            weights = layer_logits

        # Record history if enabled
        if self._record_history:
            self._routing_history.append(
                weights.detach().reshape(-1, self.num_experts).cpu()
            )

        return weights

    def get_layer_scalings(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Get scaling values for ALL layers at once.

        Useful for analysis and visualization of how the router
        distributes work across layers.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)

        Returns:
            All layer scalings: (batch_size, seq_len, num_layers, num_experts)
        """
        batch_size, seq_len, _ = hidden_states.shape
        logits = self.classifier(hidden_states)

        if self.use_layerwise:
            logits = logits.reshape(batch_size, seq_len, self.num_layers, self.num_experts)
            logits = logits + self.scaling_bias.unsqueeze(0).unsqueeze(0)
        else:
            logits = logits.unsqueeze(2).expand(-1, -1, self.num_layers, -1)

        if self.enable_softmax:
            weights = F.softmax(logits / self.temperature, dim=-1)
        else:
            weights = logits

        return weights
