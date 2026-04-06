"""Abstract base class for adapter merging strategies."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import Tensor

from .utils import load_adapter_weights, uniform_weights

logger = logging.getLogger(__name__)


class AdapterMerger(ABC):
    """Abstract base for all static adapter merging strategies.

    Subclasses implement the ``merge`` method with their specific algorithm.
    All other I/O (loading weights, saving output) is handled here.
    """

    def load_weights(
        self, adapter_paths: List[str | Path]
    ) -> List[Dict[str, Tensor]]:
        """Load A/B weight tensors from multiple adapter directories.

        Args:
            adapter_paths: List of paths to adapter directories.

        Returns:
            List of weight dicts, one per adapter.
        """
        return [load_adapter_weights(p) for p in adapter_paths]

    @abstractmethod
    def merge(
        self,
        weights_per_adapter: List[Dict[str, Tensor]],
        adapter_scalars: List[float],
    ) -> Dict[str, Tensor]:
        """Merge weights from multiple adapters into a single weight dict.

        Args:
            weights_per_adapter: List of weight dicts (one per adapter).
            adapter_scalars: Per-adapter blending coefficients (sum to 1).

        Returns:
            Merged weight dict with the same keys as the inputs.
        """

    def save(
        self,
        merged_weights: Dict[str, Tensor],
        source_config_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """Save merged weights as a standard PEFT adapter directory.

        Copies adapter_config.json from the source adapter and writes
        adapter_model.safetensors (or .bin if safetensors unavailable).

        Args:
            merged_weights: Merged weight tensors to save.
            source_config_path: Path to any source adapter directory
                (its adapter_config.json is reused).
            output_path: Directory to write the merged adapter into.

        Returns:
            Path to the output directory.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy adapter_config.json from source
        src_config = Path(source_config_path) / "adapter_config.json"
        if src_config.exists():
            with open(src_config) as f:
                config_data = json.load(f)
            with open(output_path / "adapter_config.json", "w") as f:
                json.dump(config_data, f, indent=2)

        # Save weights — prefer safetensors
        try:
            from safetensors.torch import save_file

            save_file(merged_weights, str(output_path / "adapter_model.safetensors"))
            logger.info(f"Saved merged adapter to {output_path} (safetensors)")
        except ImportError:
            torch.save(merged_weights, str(output_path / "adapter_model.bin"))
            logger.info(f"Saved merged adapter to {output_path} (bin)")

        return output_path

    def run(
        self,
        adapter_paths: List[str | Path],
        output_path: str | Path,
        weights: Optional[List[float]] = None,
        **kwargs,
    ) -> Path:
        """Full pipeline: load → merge → save.

        Args:
            adapter_paths: List of adapter directories.
            output_path: Where to write the merged adapter.
            weights: Per-adapter blending coefficients. Defaults to uniform.
            **kwargs: Passed to subclass ``merge`` (e.g. density, trim_fraction).

        Returns:
            Path to the saved merged adapter directory.
        """
        if weights is None:
            weights = uniform_weights(len(adapter_paths))

        if abs(sum(weights) - 1.0) > 1e-6:
            total = sum(weights)
            weights = [w / total for w in weights]

        all_weights = self.load_weights(adapter_paths)
        merged = self.merge(all_weights, weights, **kwargs)
        return self.save(merged, adapter_paths[0], output_path)
