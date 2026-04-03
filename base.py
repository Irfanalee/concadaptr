"""
Base router class for ConcAdptr.

All routing strategies inherit from BaseRouter and implement the forward method
to produce expert weights for each input token.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseRouter(ABC, nn.Module):
    """Abstract base class for all ConcAdptr routers.

    A router takes hidden states from the transformer and produces
    per-expert weights that determine how LoRA adapter outputs are combined.

    Args:
        hidden_size: Hidden dimension of the base transformer model.
        num_experts: Number of LoRA adapters (experts) to route across.
        num_layers: Number of transformer layers (for layer-wise routing).
        use_layerwise: Whether to produce separate routing weights per layer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_layers: int = 1,
        use_layerwise: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.use_layerwise = use_layerwise

        # Track routing statistics for analysis
        self._routing_history: list = []
        self._record_history = False

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute routing weights for each expert.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size).
            layer_idx: Current transformer layer index (for layer-wise routing).

        Returns:
            Routing weights of shape (batch_size, seq_len, num_experts).
            Weights should be non-negative and sum to 1 along the expert dimension
            (for soft merging) or be sparse (for top-k routing).
        """
        ...

    def compute_load_balance_loss(self, routing_weights: torch.Tensor) -> torch.Tensor:
        """Compute the auxiliary load-balancing loss.

        Encourages the router to distribute tokens evenly across experts,
        preventing expert collapse (where all tokens route to the same expert).

        Uses the approach from Switch Transformer (Fedus et al., 2022):
        L_balance = num_experts * sum_i(f_i * P_i)
        where f_i is the fraction of tokens routed to expert i,
        and P_i is the average routing probability for expert i.

        Args:
            routing_weights: Routing weights of shape (batch, seq_len, num_experts).

        Returns:
            Scalar load-balancing loss.
        """
        # f_i: fraction of tokens assigned to each expert
        # For soft routing, use the argmax assignment
        if routing_weights.dim() == 3:
            # (batch, seq_len, num_experts) -> flatten batch and seq
            flat_weights = routing_weights.reshape(-1, self.num_experts)
        else:
            flat_weights = routing_weights

        # Fraction of tokens where each expert has highest weight
        expert_assignments = flat_weights.argmax(dim=-1)  # (num_tokens,)
        f_i = torch.zeros(self.num_experts, device=routing_weights.device)
        for i in range(self.num_experts):
            f_i[i] = (expert_assignments == i).float().mean()

        # Average routing probability per expert
        p_i = flat_weights.mean(dim=0)  # (num_experts,)

        # Load balance loss
        loss = self.num_experts * (f_i * p_i).sum()

        return loss

    def enable_history(self, enable: bool = True) -> None:
        """Enable or disable recording of routing decisions for analysis.

        Args:
            enable: Whether to record routing history.
        """
        self._record_history = enable
        if not enable:
            self._routing_history.clear()

    def get_routing_stats(self) -> Dict[str, torch.Tensor]:
        """Get statistics about routing decisions.

        Returns:
            Dictionary with routing statistics including per-expert load,
            average routing entropy, and expert utilization.
        """
        if not self._routing_history:
            return {}

        all_weights = torch.cat(self._routing_history, dim=0)  # (total_tokens, num_experts)

        # Per-expert average load
        expert_load = all_weights.mean(dim=0)

        # Routing entropy (higher = more uniform distribution)
        eps = 1e-8
        entropy = -(all_weights * (all_weights + eps).log()).sum(dim=-1).mean()

        # Expert utilization (fraction of tokens where expert is in top-2)
        top2 = all_weights.topk(min(2, self.num_experts), dim=-1).indices
        utilization = torch.zeros(self.num_experts)
        for i in range(self.num_experts):
            utilization[i] = (top2 == i).any(dim=-1).float().mean()

        return {
            "expert_load": expert_load,
            "routing_entropy": entropy,
            "expert_utilization": utilization,
        }

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, "
            f"num_experts={self.num_experts}, "
            f"num_layers={self.num_layers}, "
            f"layerwise={self.use_layerwise}"
        )
