"""
Adapter Registry — Manage, load, and validate multiple LoRA adapters.

The registry provides a unified interface for working with collections
of independently trained LoRA adapters. It handles validation, compatibility
checking, and lazy loading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Metadata about a registered LoRA adapter.

    Args:
        name: Human-readable name (e.g., "medical", "customer_a").
        path: Filesystem path to the adapter directory.
        base_model: The base model this adapter was trained on.
        rank: LoRA rank (r parameter).
        alpha: LoRA alpha scaling parameter.
        target_modules: Which model modules this adapter targets.
        loaded: Whether the adapter weights are currently in memory.
        metadata: Additional user-defined metadata.
    """

    name: str
    path: str
    base_model: str = ""
    rank: int = 0
    alpha: int = 0
    target_modules: List[str] = field(default_factory=list)
    loaded: bool = False
    metadata: Dict = field(default_factory=dict)


class AdapterRegistry:
    """Registry for managing multiple LoRA adapters.

    Provides adapter discovery, validation, compatibility checking,
    and metadata tracking. Works with standard PEFT adapter format.

    Example:
        >>> registry = AdapterRegistry()
        >>> registry.register("medical", "./adapters/medical")
        >>> registry.register("legal", "./adapters/legal")
        >>> registry.validate_compatibility()
        True
        >>> registry.summary()
    """

    def __init__(self):
        self._adapters: Dict[str, AdapterInfo] = {}

    def register(
        self,
        name: str,
        path: Union[str, Path],
        metadata: Optional[Dict] = None,
    ) -> AdapterInfo:
        """Register a LoRA adapter with the registry.

        Args:
            name: Unique name for this adapter.
            path: Path to the adapter directory (containing adapter_config.json).
            metadata: Optional additional metadata (e.g., customer_id, domain, version).

        Returns:
            AdapterInfo with parsed adapter details.

        Raises:
            FileNotFoundError: If the adapter path doesn't exist.
            ValueError: If the adapter name is already registered.
        """
        if name in self._adapters:
            raise ValueError(
                f"Adapter '{name}' is already registered. "
                f"Use unregister('{name}') first or choose a different name."
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {path}")

        # Parse adapter_config.json
        config_path = path / "adapter_config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No adapter_config.json found in {path}. "
                f"Is this a valid PEFT adapter directory?"
            )

        with open(config_path) as f:
            config = json.load(f)

        info = AdapterInfo(
            name=name,
            path=str(path),
            base_model=config.get("base_model_name_or_path", ""),
            rank=config.get("r", 0),
            alpha=config.get("lora_alpha", 0),
            target_modules=config.get("target_modules", []),
            metadata=metadata or {},
        )

        # Check for weight files
        has_safetensors = (path / "adapter_model.safetensors").exists()
        has_bin = (path / "adapter_model.bin").exists()
        if not has_safetensors and not has_bin:
            logger.warning(
                f"No adapter weights found in {path}. "
                f"Expected adapter_model.safetensors or adapter_model.bin."
            )

        self._adapters[name] = info
        logger.info(
            f"Registered adapter '{name}': rank={info.rank}, "
            f"alpha={info.alpha}, modules={info.target_modules}"
        )

        return info

    def register_from_dict(
        self, adapters: Dict[str, str], metadata: Optional[Dict[str, Dict]] = None
    ) -> List[AdapterInfo]:
        """Register multiple adapters from a name→path dictionary.

        Args:
            adapters: Mapping of adapter names to paths.
            metadata: Optional per-adapter metadata.

        Returns:
            List of registered AdapterInfo objects.
        """
        results = []
        for name, path in adapters.items():
            meta = (metadata or {}).get(name)
            results.append(self.register(name, path, metadata=meta))
        return results

    def unregister(self, name: str) -> None:
        """Remove an adapter from the registry.

        Args:
            name: Name of the adapter to remove.
        """
        if name not in self._adapters:
            raise KeyError(f"Adapter '{name}' is not registered.")
        del self._adapters[name]
        logger.info(f"Unregistered adapter '{name}'")

    def get(self, name: str) -> AdapterInfo:
        """Get adapter info by name.

        Args:
            name: Adapter name.

        Returns:
            AdapterInfo for the requested adapter.
        """
        if name not in self._adapters:
            raise KeyError(
                f"Adapter '{name}' not found. "
                f"Available: {list(self._adapters.keys())}"
            )
        return self._adapters[name]

    @property
    def names(self) -> List[str]:
        """List of all registered adapter names."""
        return list(self._adapters.keys())

    @property
    def num_adapters(self) -> int:
        """Number of registered adapters."""
        return len(self._adapters)

    def validate_compatibility(self) -> bool:
        """Check that all registered adapters are compatible for fusion.

        Compatibility requires:
        - Same base model (or all empty, meaning unspecified)
        - Same LoRA rank
        - Same target modules

        Returns:
            True if all adapters are compatible.

        Raises:
            ValueError: If adapters are incompatible, with a detailed message.
        """
        if len(self._adapters) < 2:
            return True

        adapters = list(self._adapters.values())
        reference = adapters[0]
        issues = []

        for adapter in adapters[1:]:
            # Check base model
            if (
                reference.base_model
                and adapter.base_model
                and reference.base_model != adapter.base_model
            ):
                issues.append(
                    f"Base model mismatch: '{reference.name}' uses "
                    f"'{reference.base_model}' but '{adapter.name}' uses "
                    f"'{adapter.base_model}'"
                )

            # Check rank
            if reference.rank != adapter.rank:
                issues.append(
                    f"Rank mismatch: '{reference.name}' has rank={reference.rank} "
                    f"but '{adapter.name}' has rank={adapter.rank}"
                )

            # Check target modules
            if set(reference.target_modules) != set(adapter.target_modules):
                issues.append(
                    f"Target module mismatch: '{reference.name}' targets "
                    f"{reference.target_modules} but '{adapter.name}' targets "
                    f"{adapter.target_modules}"
                )

        if issues:
            raise ValueError(
                "Adapter compatibility check failed:\n" + "\n".join(f"  - {i}" for i in issues)
            )

        return True

    def summary(self) -> str:
        """Generate a human-readable summary of all registered adapters.

        Returns:
            Formatted summary string.
        """
        if not self._adapters:
            return "No adapters registered."

        lines = [
            f"ConcAdpt Adapter Registry ({self.num_adapters} adapters)",
            "=" * 60,
        ]

        for name, info in self._adapters.items():
            lines.append(f"\n  [{name}]")
            lines.append(f"    Path:    {info.path}")
            lines.append(f"    Base:    {info.base_model or '(unspecified)'}")
            lines.append(f"    Rank:    {info.rank}")
            lines.append(f"    Alpha:   {info.alpha}")
            lines.append(f"    Modules: {', '.join(info.target_modules) or '(unspecified)'}")
            lines.append(f"    Loaded:  {info.loaded}")
            if info.metadata:
                lines.append(f"    Meta:    {info.metadata}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"AdapterRegistry(adapters={self.names})"

    def __len__(self) -> int:
        return self.num_adapters

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __iter__(self):
        return iter(self._adapters.values())
