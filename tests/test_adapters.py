"""Tests for AdapterRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from concadptr.adapters import AdapterRegistry


class TestAdapterRegistry:
    def test_register_adapter(self, adapter_dir):
        registry = AdapterRegistry()
        info = registry.register("medical", adapter_dir("medical"))
        assert info.name == "medical"
        assert info.rank == 16
        assert registry.num_adapters == 1

    def test_register_duplicate_fails(self, adapter_dir):
        registry = AdapterRegistry()
        d = adapter_dir("medical")
        registry.register("medical", d)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("medical", d)

    def test_register_missing_path_raises(self, tmp_path):
        registry = AdapterRegistry()
        with pytest.raises(FileNotFoundError):
            registry.register("x", tmp_path / "nonexistent")

    def test_register_missing_config_raises(self, tmp_path):
        d = tmp_path / "no_config"
        d.mkdir()
        # weights file present but no adapter_config.json
        torch.save({}, d / "adapter_model.bin")
        registry = AdapterRegistry()
        with pytest.raises(FileNotFoundError, match="adapter_config.json"):
            registry.register("x", d)

    def test_compatibility_check_passes(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a", rank=16))
        registry.register("b", adapter_dir("b", rank=16))
        assert registry.validate_compatibility() is True

    def test_compatibility_check_fails_rank_mismatch(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a", rank=16))
        registry.register("b", adapter_dir("b", rank=32))
        with pytest.raises(ValueError, match="Rank mismatch"):
            registry.validate_compatibility()

    def test_compatibility_check_fails_base_model_mismatch(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a", base_model="model-a"))
        registry.register("b", adapter_dir("b", base_model="model-b"))
        with pytest.raises(ValueError, match="Base model mismatch"):
            registry.validate_compatibility()

    def test_register_from_dict(self, adapter_dir):
        dir_a = adapter_dir("a")
        dir_b = adapter_dir("b")
        registry = AdapterRegistry()
        registry.register_from_dict({"a": str(dir_a), "b": str(dir_b)})
        assert registry.num_adapters == 2

    def test_names_property(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("x", adapter_dir("x"))
        registry.register("y", adapter_dir("y"))
        assert set(registry.names) == {"x", "y"}

    def test_unregister(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a"))
        registry.unregister("a")
        assert registry.num_adapters == 0

    def test_unregister_nonexistent_raises(self):
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_by_name(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("medical", adapter_dir("medical"))
        info = registry.get("medical")
        assert info.name == "medical"

    def test_get_missing_raises(self):
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.get("missing")

    def test_contains(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a"))
        assert "a" in registry
        assert "b" not in registry

    def test_len(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a"))
        registry.register("b", adapter_dir("b"))
        assert len(registry) == 2

    def test_iter(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a"))
        registry.register("b", adapter_dir("b"))
        names = [info.name for info in registry]
        assert set(names) == {"a", "b"}

    def test_summary_contains_adapter_name(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("medical", adapter_dir("medical"))
        summary = registry.summary()
        assert "medical" in summary
        assert "rank" in summary.lower() or "Rank" in summary

    def test_metadata_stored(self, adapter_dir):
        registry = AdapterRegistry()
        meta = {"customer_id": "acme", "version": "1.0"}
        info = registry.register("a", adapter_dir("a"), metadata=meta)
        assert info.metadata["customer_id"] == "acme"

    def test_single_adapter_passes_compatibility(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("solo", adapter_dir("solo"))
        assert registry.validate_compatibility() is True
