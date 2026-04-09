"""Static adapter merging — Linear, TIES, DARE, DARE+TIES.

Merges independently trained LoRA adapters into a single adapter using
static (non-routing) techniques. The output is a standard PEFT adapter
directory loadable by ``PeftModel.from_pretrained()``.

Example:
    >>> from concadptr.merging import merge_adapters
    >>> output = merge_adapters(
    ...     adapters={"medical": "/path/a", "legal": "/path/b"},
    ...     output_path="./merged",
    ...     method="ties",
    ...     weights=[0.6, 0.4],
    ...     trim_fraction=0.2,
    ... )
"""

from __future__ import annotations

from pathlib import Path

from .base import AdapterMerger
from .dare import DAREMerge
from .linear import LinearMerge
from .progressive import MergeResult, ProgressiveMerger, ProgressiveMergerConfig, QualityGateError
from .ties import TIESMerge

__all__ = [
    "merge_adapters",
    "AdapterMerger",
    "LinearMerge",
    "TIESMerge",
    "DAREMerge",
    "ProgressiveMerger",
    "ProgressiveMergerConfig",
    "MergeResult",
    "QualityGateError",
]


def merge_adapters(
    adapters: dict[str, str],
    output_path: str | Path,
    method: str = "linear",
    weights: list[float] | None = None,
    density: float = 0.7,
    trim_fraction: float = 0.2,
    seed: int = 42,
) -> Path:
    """Merge multiple LoRA adapters into a single PEFT adapter directory.

    Args:
        adapters: Mapping of adapter names to their directory paths.
            Order determines pairing with ``weights``.
        output_path: Directory to write the merged adapter.
        method: Merging strategy — one of ``"linear"``, ``"ties"``,
            ``"dare"``, ``"dare_ties"``.
        weights: Per-adapter blending coefficients. Must match the number
            of adapters. Defaults to uniform weights.
        density: DARE only — fraction of parameters to keep (0 < density ≤ 1).
        trim_fraction: TIES only — fraction of low-magnitude params to zero out.
        seed: DARE random seed for reproducible dropout masks.

    Returns:
        Path to the saved merged adapter directory.

    Raises:
        ValueError: If an unknown method is specified.
    """
    adapter_paths = list(adapters.values())

    merger: AdapterMerger
    if method == "linear":
        merger = LinearMerge()
        return merger.run(adapter_paths, output_path, weights=weights)
    elif method == "ties":
        merger = TIESMerge()
        return merger.run(adapter_paths, output_path, weights=weights, trim_fraction=trim_fraction)
    elif method == "dare":
        merger = DAREMerge(use_ties=False)
        return merger.run(adapter_paths, output_path, weights=weights, density=density, seed=seed)
    elif method == "dare_ties":
        merger = DAREMerge(use_ties=True)
        return merger.run(
            adapter_paths, output_path, weights=weights,
            density=density, seed=seed, trim_fraction=trim_fraction,
        )
    else:
        raise ValueError(
            f"Unknown merge method: '{method}'. "
            "Choose from: 'linear', 'ties', 'dare', 'dare_ties'."
        )
