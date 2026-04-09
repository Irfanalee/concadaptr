"""
Adapter Registry — Manage, load, and validate multiple LoRA adapters.

The registry provides a unified interface for working with collections
of independently trained LoRA adapters. It handles validation, compatibility
checking, and lazy loading.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_VERSION_FILE = "concadptr_version.json"


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
        hub_repo_id: HuggingFace Hub repo ID if loaded from Hub.
        version: Semantic version string set by the user (e.g. "1.2.0").
        created_at: ISO 8601 timestamp of when the adapter was produced.
        training_config_hash: SHA-256 of the training configuration dict,
            computed via AdapterInfo.compute_config_hash().
        eval_metrics: Task-specific evaluation results, e.g.
            {"mmlu_accuracy": 0.72, "rouge1": 0.45}.
    """

    name: str
    path: str
    base_model: str = ""
    rank: int = 0
    alpha: int = 0
    target_modules: list[str] = field(default_factory=list)
    loaded: bool = False
    metadata: dict = field(default_factory=dict)
    hub_repo_id: str | None = None

    # Version metadata (§7.3)
    version: str | None = None
    created_at: str | None = None
    training_config_hash: str | None = None
    eval_metrics: dict[str, float] = field(default_factory=dict)

    @staticmethod
    def compute_config_hash(config) -> str:
        """Compute a deterministic SHA-256 hash of a training configuration.

        Accepts either a plain dict or any dataclass instance (e.g.
        ``TrainingConfig``). Keys are sorted before hashing so field order
        does not affect the result.

        Args:
            config: A dict or dataclass instance representing the training config.

        Returns:
            64-character lowercase hex SHA-256 digest.

        Raises:
            TypeError: If config is not a dict or dataclass instance.
        """
        if dataclasses.is_dataclass(config) and not isinstance(config, type):
            config_dict = dataclasses.asdict(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(
                f"Expected a dict or dataclass instance, got {type(config).__name__}"
            )
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def save_version_metadata(self, path: str | Path | None = None) -> Path:
        """Write version metadata to ``concadptr_version.json``.

        Only non-None / non-empty fields are written, so the file stays
        minimal. Existing keys not present on this AdapterInfo are
        preserved.

        Args:
            path: Override save location. Defaults to
                ``{self.path}/concadptr_version.json``.

        Returns:
            Path to the saved file.
        """
        target = Path(path) if path is not None else Path(self.path) / _VERSION_FILE

        # Preserve any unknown keys already in the file
        existing: dict = {}
        if target.exists():
            with open(target) as f:
                existing = json.load(f)

        if self.version is not None:
            existing["version"] = self.version
        if self.created_at is not None:
            existing["created_at"] = self.created_at
        if self.training_config_hash is not None:
            existing["training_config_hash"] = self.training_config_hash
        if self.eval_metrics:
            existing["eval_metrics"] = self.eval_metrics

        with open(target, "w") as f:
            json.dump(existing, f, indent=2)

        logger.debug(f"Version metadata saved to {target}")
        return target

    @classmethod
    def _load_version_fields(cls, path: Path) -> dict:
        """Read version metadata from a concadptr_version.json file.

        Args:
            path: Adapter directory containing the version file.

        Returns:
            Dict with keys version, created_at, training_config_hash,
            eval_metrics (all optional). Empty dict if the file is absent.
        """
        version_file = path / _VERSION_FILE
        if not version_file.exists():
            return {}
        with open(version_file) as f:
            data = json.load(f)
        return {
            "version": data.get("version"),
            "created_at": data.get("created_at"),
            "training_config_hash": data.get("training_config_hash"),
            "eval_metrics": data.get("eval_metrics", {}),
        }


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
        self._adapters: dict[str, AdapterInfo] = {}

    def register(
        self,
        name: str,
        path: str | Path,
        metadata: dict | None = None,
    ) -> AdapterInfo:
        """Register a LoRA adapter with the registry.

        If a ``concadptr_version.json`` file exists in the adapter directory,
        its version, created_at, training_config_hash, and eval_metrics are
        loaded automatically.

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

        # Load version metadata if present
        ver = AdapterInfo._load_version_fields(path)

        info = AdapterInfo(
            name=name,
            path=str(path),
            base_model=config.get("base_model_name_or_path", ""),
            rank=config.get("r", 0),
            alpha=config.get("lora_alpha", 0),
            target_modules=config.get("target_modules", []),
            metadata=metadata or {},
            version=ver.get("version"),
            created_at=ver.get("created_at"),
            training_config_hash=ver.get("training_config_hash"),
            eval_metrics=ver.get("eval_metrics", {}),
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
        self, adapters: dict[str, str], metadata: dict[str, dict] | None = None
    ) -> list[AdapterInfo]:
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

    def set_eval_metrics(
        self,
        name: str,
        metrics: dict[str, float],
        save: bool = True,
    ) -> None:
        """Update the evaluation metrics for a registered adapter.

        Merges new metrics into the existing eval_metrics dict (does not
        replace). Optionally persists them to ``concadptr_version.json``
        in the adapter directory so they survive re-registration.

        Args:
            name: Name of the registered adapter.
            metrics: Metric name → value pairs to merge in.
                E.g. {"mmlu_accuracy": 0.72, "rouge1": 0.45}.
            save: If True, write updated metadata to disk immediately.
        """
        info = self.get(name)
        info.eval_metrics.update(metrics)
        if save:
            info.save_version_metadata()
        logger.info(f"Updated eval metrics for '{name}': {metrics}")

    @property
    def names(self) -> list[str]:
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
            f"ConcAdptr Adapter Registry ({self.num_adapters} adapters)",
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
            if info.version:
                lines.append(f"    Version: {info.version}")
            if info.created_at:
                lines.append(f"    Created: {info.created_at}")
            if info.training_config_hash:
                lines.append(f"    Config:  {info.training_config_hash[:16]}...")
            if info.eval_metrics:
                metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in info.eval_metrics.items())
                lines.append(f"    Metrics: {metrics_str}")
            if info.metadata:
                lines.append(f"    Meta:    {info.metadata}")

        return "\n".join(lines)

    def push_adapter_to_hub(
        self,
        name: str,
        repo_id: str,
        token: str | None = None,
        private: bool = False,
        commit_message: str | None = None,
    ) -> str:
        """Upload an adapter to the HuggingFace Hub.

        The adapter directory (adapter_config.json + weights +
        concadptr_version.json if present) is uploaded as a standard PEFT
        adapter repo.

        Args:
            name: Name of the registered adapter to push.
            repo_id: Hub repo ID, e.g. "username/my-adapter".
            token: HF token (uses cached login if None).
            private: Create a private repo.
            commit_message: Custom commit message.

        Returns:
            URL of the uploaded repo.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hub operations. "
                "Install it with: pip install 'concadptr[hub]'"
            )

        info = self.get(name)
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        url = api.upload_folder(
            folder_path=info.path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message or f"Upload adapter '{name}' via concadptr",
        )
        logger.info(f"Adapter '{name}' pushed to {repo_id}")
        return url

    def load_adapter_from_hub(
        self,
        repo_id: str,
        name: str | None = None,
        token: str | None = None,
        cache_dir: str | None = None,
    ) -> AdapterInfo:
        """Download an adapter from the HuggingFace Hub and register it.

        Args:
            repo_id: Hub repo ID, e.g. "username/my-adapter".
            name: Local name for the adapter. Defaults to the repo name portion.
            token: HF token (uses cached login if None).
            cache_dir: Local directory to cache downloaded files.

        Returns:
            AdapterInfo for the registered adapter.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is required for Hub operations. "
                "Install it with: pip install 'concadptr[hub]'"
            )

        if name is None:
            name = repo_id.split("/")[-1]

        local_path = snapshot_download(
            repo_id=repo_id,
            token=token,
            cache_dir=cache_dir,
        )
        info = self.register(name, local_path)
        info.hub_repo_id = repo_id
        logger.info(f"Adapter '{name}' loaded from {repo_id} → {local_path}")
        return info

    def merge(
        self,
        adapter_names: list[str],
        output_path: str | Path,
        method: str = "linear",
        **kwargs,
    ) -> Path:
        """Merge registered adapters into a single PEFT adapter directory.

        Validates adapter compatibility before merging.

        Args:
            adapter_names: Names of registered adapters to merge.
            output_path: Directory to write the merged adapter.
            method: One of ``"linear"``, ``"ties"``, ``"dare"``, ``"dare_ties"``.
            **kwargs: Forwarded to ``merge_adapters`` (e.g. weights, density,
                trim_fraction, seed).

        Returns:
            Path to the saved merged adapter directory.
        """
        from concadptr.merging import merge_adapters

        # Validate that all requested adapters are registered
        for name in adapter_names:
            self.get(name)  # raises KeyError if missing

        # Temporarily restrict registry to the requested adapters and validate
        subset_adapters = {name: self._adapters[name] for name in adapter_names}
        original = self._adapters
        self._adapters = subset_adapters
        try:
            self.validate_compatibility()
        finally:
            self._adapters = original

        adapter_paths = {name: self._adapters[name].path for name in adapter_names}
        return merge_adapters(adapter_paths, output_path, method=method, **kwargs)

    def __repr__(self) -> str:
        return f"AdapterRegistry(adapters={self.names})"

    def __len__(self) -> int:
        return self.num_adapters

    def __contains__(self, name: str) -> bool:
        return name in self._adapters

    def __iter__(self):
        return iter(self._adapters.values())
