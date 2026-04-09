"""Tests for AdapterRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from concadptr.adapters import AdapterInfo, AdapterRegistry


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


class TestAdapterVersionMetadata:
    def test_new_fields_default_to_none_or_empty(self, adapter_dir):
        registry = AdapterRegistry()
        info = registry.register("a", adapter_dir("a"))
        assert info.version is None
        assert info.created_at is None
        assert info.training_config_hash is None
        assert info.eval_metrics == {}

    def test_compute_config_hash_is_deterministic(self):
        config = {"learning_rate": 1e-4, "num_epochs": 3, "batch_size": 4}
        h1 = AdapterInfo.compute_config_hash(config)
        h2 = AdapterInfo.compute_config_hash(config)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_config_hash_differs_for_different_configs(self):
        h1 = AdapterInfo.compute_config_hash({"lr": 1e-4})
        h2 = AdapterInfo.compute_config_hash({"lr": 1e-3})
        assert h1 != h2

    def test_compute_config_hash_accepts_dataclass(self):
        from dataclasses import dataclass

        @dataclass
        class FakeConfig:
            lr: float = 1e-4
            epochs: int = 3

        h = AdapterInfo.compute_config_hash(FakeConfig())
        assert isinstance(h, str) and len(h) == 64

    def test_compute_config_hash_raises_for_invalid_type(self):
        with pytest.raises(TypeError):
            AdapterInfo.compute_config_hash("not_a_dict_or_dataclass")

    def test_save_version_metadata_creates_file(self, adapter_dir, tmp_path):
        path = adapter_dir("a")
        registry = AdapterRegistry()
        info = registry.register("a", path)
        info.version = "1.0.0"
        info.eval_metrics = {"mmlu": 0.72}
        info.save_version_metadata()

        version_file = path / "concadptr_version.json"
        assert version_file.exists()
        data = json.loads(version_file.read_text())
        assert data["version"] == "1.0.0"
        assert data["eval_metrics"]["mmlu"] == pytest.approx(0.72)

    def test_save_version_metadata_custom_path(self, adapter_dir, tmp_path):
        path = adapter_dir("a")
        registry = AdapterRegistry()
        info = registry.register("a", path)
        info.version = "2.0.0"
        target = tmp_path / "ver.json"
        info.save_version_metadata(path=target)
        assert target.exists()

    def test_save_version_metadata_preserves_existing_keys(self, adapter_dir):
        path = adapter_dir("a")
        version_file = path / "concadptr_version.json"
        version_file.write_text(json.dumps({"custom_key": "kept"}))

        registry = AdapterRegistry()
        info = registry.register("a", path)
        info.version = "1.1.0"
        info.save_version_metadata()

        data = json.loads(version_file.read_text())
        assert data["custom_key"] == "kept"
        assert data["version"] == "1.1.0"

    def test_register_loads_version_metadata_automatically(self, adapter_dir):
        path = adapter_dir("a")
        version_file = path / "concadptr_version.json"
        version_file.write_text(
            json.dumps({
                "version": "3.0.0",
                "created_at": "2026-04-09T00:00:00",
                "training_config_hash": "abc123",
                "eval_metrics": {"accuracy": 0.88},
            })
        )

        registry = AdapterRegistry()
        info = registry.register("a", path)

        assert info.version == "3.0.0"
        assert info.created_at == "2026-04-09T00:00:00"
        assert info.training_config_hash == "abc123"
        assert info.eval_metrics["accuracy"] == pytest.approx(0.88)

    def test_register_without_version_file_still_works(self, adapter_dir):
        path = adapter_dir("a")
        registry = AdapterRegistry()
        info = registry.register("a", path)
        assert info.version is None
        assert info.eval_metrics == {}

    def test_set_eval_metrics_updates_info(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a"))
        registry.set_eval_metrics("a", {"mmlu": 0.70, "hellaswag": 0.75}, save=False)
        info = registry.get("a")
        assert info.eval_metrics["mmlu"] == pytest.approx(0.70)
        assert info.eval_metrics["hellaswag"] == pytest.approx(0.75)

    def test_set_eval_metrics_merges_not_replaces(self, adapter_dir):
        registry = AdapterRegistry()
        registry.register("a", adapter_dir("a"))
        registry.set_eval_metrics("a", {"mmlu": 0.70}, save=False)
        registry.set_eval_metrics("a", {"hellaswag": 0.80}, save=False)
        info = registry.get("a")
        assert "mmlu" in info.eval_metrics
        assert "hellaswag" in info.eval_metrics

    def test_set_eval_metrics_saves_to_disk(self, adapter_dir):
        path = adapter_dir("a")
        registry = AdapterRegistry()
        registry.register("a", path)
        registry.set_eval_metrics("a", {"accuracy": 0.65}, save=True)
        version_file = path / "concadptr_version.json"
        assert version_file.exists()
        data = json.loads(version_file.read_text())
        assert data["eval_metrics"]["accuracy"] == pytest.approx(0.65)

    def test_set_eval_metrics_raises_for_unknown_adapter(self):
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.set_eval_metrics("nonexistent", {"acc": 0.5})

    def test_summary_includes_version_and_metrics(self, adapter_dir):
        path = adapter_dir("a")
        registry = AdapterRegistry()
        info = registry.register("a", path)
        info.version = "1.0.0"
        info.eval_metrics = {"accuracy": 0.72}
        summary = registry.summary()
        assert "1.0.0" in summary
        assert "accuracy" in summary
