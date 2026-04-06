"""DARE (Drop And REscale) adapter merging.

Reference: "Language Models are Super Mario: Absorbing Abilities from
Homologous Models as a Free Lunch" (Yu et al., 2023).
https://arxiv.org/abs/2311.03099
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import Tensor

from .base import AdapterMerger
from .linear import LinearMerge
from .ties import TIESMerge


class DAREMerge(AdapterMerger):
    """Merge adapters using the DARE algorithm.

    For each adapter:
    1. **Drop**: Randomly zero out parameters with probability ``(1 - density)``.
    2. **Rescale**: Multiply surviving parameters by ``1 / density`` to preserve
       expected magnitude.

    After sparsification, the adapters are merged using either linear averaging
    (default) or TIES (when ``use_ties=True``).

    Example:
        >>> merger = DAREMerge()
        >>> output = merger.run(
        ...     adapter_paths=["/path/a", "/path/b"],
        ...     output_path="./merged",
        ...     density=0.7,
        ...     seed=42,
        ... )

        >>> # DARE + TIES combined
        >>> merger = DAREMerge(use_ties=True)
        >>> output = merger.run(
        ...     adapter_paths=["/path/a", "/path/b"],
        ...     output_path="./merged",
        ...     density=0.7,
        ...     trim_fraction=0.2,
        ... )
    """

    def __init__(self, use_ties: bool = False):
        """Initialize DAREMerge.

        Args:
            use_ties: If True, apply TIES merging after DARE sparsification
                instead of plain linear merging.
        """
        self.use_ties = use_ties
        self._linear = LinearMerge()
        self._ties = TIESMerge()

    def merge(
        self,
        weights_per_adapter: List[Dict[str, Tensor]],
        adapter_scalars: List[float],
        density: float = 0.7,
        seed: int = 42,
        trim_fraction: float = 0.2,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """DARE merging algorithm.

        Args:
            weights_per_adapter: List of weight dicts (one per adapter).
            adapter_scalars: Per-adapter blending coefficients (sum to 1).
            density: Fraction of parameters to KEEP per adapter (0 < density ≤ 1).
            seed: Random seed for reproducible dropout masks.
            trim_fraction: TIES trim fraction (only used when ``use_ties=True``).

        Returns:
            Merged weight dict.
        """
        if not (0.0 < density <= 1.0):
            raise ValueError(f"density must be in (0, 1], got {density}")

        rng = torch.Generator()
        rng.manual_seed(seed)

        sparsified = []
        for adapter in weights_per_adapter:
            sparse_adapter: Dict[str, Tensor] = {}
            for key, tensor in adapter.items():
                t = tensor.float()
                mask = torch.bernoulli(
                    torch.full(t.shape, density), generator=rng
                ).bool()
                sparse_adapter[key] = (t * mask.float()) / density
            sparsified.append(sparse_adapter)

        if self.use_ties:
            return self._ties.merge(sparsified, adapter_scalars, trim_fraction=trim_fraction)
        return self._linear.merge(sparsified, adapter_scalars)
