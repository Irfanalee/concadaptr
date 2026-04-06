"""Utility functions for adapter merging."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor


def load_adapter_weights(path: str | Path) -> Dict[str, Tensor]:
    """Load LoRA A/B weight tensors from an adapter directory.

    Supports both .safetensors (preferred) and .bin formats.

    Args:
        path: Path to the adapter directory.

    Returns:
        Dict mapping parameter names to tensors (on CPU).

    Raises:
        FileNotFoundError: If no weight file is found.
    """
    path = Path(path)
    safetensors_path = path / "adapter_model.safetensors"
    bin_path = path / "adapter_model.bin"

    if safetensors_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "safetensors is required to load .safetensors files. "
                "Install it with: pip install safetensors"
            )
        return load_file(str(safetensors_path), device="cpu")

    if bin_path.exists():
        return torch.load(str(bin_path), map_location="cpu")

    raise FileNotFoundError(
        f"No adapter weights found in {path}. "
        "Expected adapter_model.safetensors or adapter_model.bin."
    )


def uniform_weights(n: int) -> List[float]:
    """Return a list of n uniform weights that sum to 1.

    Args:
        n: Number of adapters.

    Returns:
        List of floats, each equal to 1/n.
    """
    return [1.0 / n] * n
