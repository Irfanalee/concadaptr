"""TIES (Trim, Elect Sign, Merge) adapter merging.

Reference: "TIES-Merging: Resolving Interference When Merging Models"
(Yadav et al., 2023). https://arxiv.org/abs/2306.01708
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch import Tensor

from .base import AdapterMerger


class TIESMerge(AdapterMerger):
    """Merge adapters using the TIES algorithm.

    Three phases:
    1. **Trim**: Zero out parameters below the ``trim_fraction``-th percentile
       of magnitude within each adapter.
    2. **Sign elect**: For each position, take the majority sign across all
       non-zero values. Ties are broken by the sign of the sum.
    3. **Merge**: Average only the parameters whose sign agrees with the
       elected sign; set all others to zero.

    Example:
        >>> merger = TIESMerge()
        >>> output = merger.run(
        ...     adapter_paths=["/path/a", "/path/b"],
        ...     output_path="./merged",
        ...     trim_fraction=0.2,
        ... )
    """

    def merge(
        self,
        weights_per_adapter: List[Dict[str, Tensor]],
        adapter_scalars: List[float],
        trim_fraction: float = 0.2,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """TIES merging algorithm.

        Args:
            weights_per_adapter: List of weight dicts (one per adapter).
            adapter_scalars: Per-adapter blending coefficients (sum to 1).
            trim_fraction: Fraction of low-magnitude parameters to zero out
                within each adapter (0.0 = no trimming, 1.0 = zero everything).

        Returns:
            Merged weight dict.
        """
        keys = weights_per_adapter[0].keys()
        merged: Dict[str, Tensor] = {}

        for key in keys:
            tensors = [adapter[key].float() for adapter in weights_per_adapter]

            # Phase 1: Trim low-magnitude parameters per adapter
            trimmed = []
            for t in tensors:
                if trim_fraction > 0.0:
                    flat = t.abs().flatten()
                    k = max(1, int(trim_fraction * flat.numel()))
                    threshold = torch.topk(flat, k, largest=False).values.max()
                    mask = t.abs() > threshold
                    trimmed.append(t * mask.float())
                else:
                    trimmed.append(t.clone())

            # Phase 2: Elect majority sign for each position
            # Stack → shape (num_adapters, *param_shape)
            stacked = torch.stack(trimmed, dim=0)
            sign_sum = stacked.sign().sum(dim=0)  # positive = mostly +, negative = mostly -
            elected_sign = sign_sum.sign()
            # Ties (sign_sum == 0): use sign of weighted sum
            tie_mask = elected_sign == 0
            if tie_mask.any():
                weighted_sum = sum(
                    s * t for s, t in zip(adapter_scalars, trimmed)
                )
                elected_sign[tie_mask] = weighted_sum.sign()[tie_mask]

            # Phase 3: Average only values that agree with elected sign
            agreement_sum = torch.zeros_like(tensors[0])
            weight_sum = torch.zeros_like(tensors[0])

            for scalar, t in zip(adapter_scalars, trimmed):
                agrees = (t.sign() == elected_sign) & (t != 0)
                agreement_sum += scalar * t * agrees.float()
                weight_sum += scalar * agrees.float()

            # Avoid division by zero
            weight_sum = weight_sum.clamp(min=1e-8)
            merged[key] = (agreement_sum / weight_sum) * elected_sign.abs()

            # Cast back to original dtype
            merged[key] = merged[key].to(weights_per_adapter[0][key].dtype)

        return merged
