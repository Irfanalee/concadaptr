"""Linear (weighted average) adapter merging."""

from __future__ import annotations

from typing import Dict, List

from torch import Tensor

from .base import AdapterMerger


class LinearMerge(AdapterMerger):
    """Merge adapters by computing a weighted average of each parameter.

    For each parameter key ``k``:
        merged[k] = Σ(w_i * adapter_i[k])

    This is the simplest merging strategy and works well when adapters
    are trained on related tasks or when equal contribution is desired.

    Example:
        >>> merger = LinearMerge()
        >>> output = merger.run(
        ...     adapter_paths=["/path/a", "/path/b"],
        ...     output_path="./merged",
        ...     weights=[0.6, 0.4],
        ... )
    """

    def merge(
        self,
        weights_per_adapter: List[Dict[str, Tensor]],
        adapter_scalars: List[float],
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Weighted average of all adapter parameters.

        Args:
            weights_per_adapter: List of weight dicts (one per adapter).
            adapter_scalars: Per-adapter blending coefficients (sum to 1).

        Returns:
            Merged weight dict.
        """
        keys = weights_per_adapter[0].keys()
        merged: Dict[str, Tensor] = {}
        for key in keys:
            merged[key] = sum(
                scalar * adapter[key]
                for scalar, adapter in zip(adapter_scalars, weights_per_adapter)
            )
        return merged
