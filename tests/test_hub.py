"""Tests for HuggingFace Hub push/pull integration.

All network calls are mocked — no real Hub access required.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from concadptr.adapters import AdapterInfo, AdapterRegistry
from concadptr.config import ConcAdptrConfig
from concadptr.model import ConcAdptrModel
from concadptr.router.soft_merging import SoftMergingRouter

from .conftest import make_mock_base_model

HIDDEN, VOCAB, NUM_EXPERTS = 64, 100, 2


def _make_registry_model(tmp_path: Path) -> ConcAdptrModel:
    """Minimal ConcAdptrModel with two fake adapters, no network calls."""
    config = ConcAdptrConfig(
        base_model="fake",
        adapters={"a": str(tmp_path / "a"), "b": str(tmp_path / "b")},
        routing_strategy="soft_merging",
        quantization=None,
    )
    model = ConcAdptrModel(config)
    model.base_model = make_mock_base_model()
    model.router = SoftMergingRouter(hidden_size=HIDDEN, num_experts=NUM_EXPERTS, num_layers=2)
    model.registry._adapters = {
        "a": AdapterInfo(name="a", path=str(tmp_path / "a"), rank=16),
        "b": AdapterInfo(name="b", path=str(tmp_path / "b"), rank=16, hub_repo_id="user/b-adapter"),
    }
    model._is_loaded = True
    return model


# ── AdapterRegistry Hub methods ──────────────────────────────────────────────


class TestPushAdapterToHub:
    def test_calls_upload_folder(self, tmp_path):
        registry = AdapterRegistry()
        registry._adapters["med"] = AdapterInfo(name="med", path=str(tmp_path), rank=16)

        mock_api = MagicMock()
        mock_api.upload_folder.return_value = "https://huggingface.co/user/med"

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            url = registry.push_adapter_to_hub("med", repo_id="user/med", token="tok")

        mock_api.create_repo.assert_called_once_with(
            repo_id="user/med", repo_type="model", private=False, exist_ok=True
        )
        mock_api.upload_folder.assert_called_once()
        call_kwargs = mock_api.upload_folder.call_args.kwargs
        assert call_kwargs["repo_id"] == "user/med"
        assert call_kwargs["folder_path"] == str(tmp_path)
        assert url == "https://huggingface.co/user/med"

    def test_raises_on_missing_adapter(self, tmp_path):
        registry = AdapterRegistry()
        with patch("huggingface_hub.HfApi", return_value=MagicMock()):
            with pytest.raises(KeyError):
                registry.push_adapter_to_hub("nonexistent", repo_id="user/x")

    def test_raises_without_huggingface_hub(self, tmp_path):
        registry = AdapterRegistry()
        registry._adapters["a"] = AdapterInfo(name="a", path=str(tmp_path), rank=8)
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="concadptr\\[hub\\]"):
                registry.push_adapter_to_hub("a", repo_id="user/a")


class TestLoadAdapterFromHub:
    def test_registers_adapter_with_hub_repo_id(self, tmp_path, adapter_dir):
        adapter_path = adapter_dir("from_hub")
        registry = AdapterRegistry()

        with patch("huggingface_hub.snapshot_download", return_value=str(adapter_path)):
            info = registry.load_adapter_from_hub("user/my-adapter", name="hub_adapter")

        assert "hub_adapter" in registry
        assert info.hub_repo_id == "user/my-adapter"
        assert info.path == str(adapter_path)

    def test_name_defaults_to_repo_suffix(self, tmp_path, adapter_dir):
        adapter_path = adapter_dir("suffix_test")
        registry = AdapterRegistry()

        with patch("huggingface_hub.snapshot_download", return_value=str(adapter_path)):
            info = registry.load_adapter_from_hub("user/my-adapter")

        assert "my-adapter" in registry

    def test_raises_without_huggingface_hub(self):
        registry = AdapterRegistry()
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="concadptr\\[hub\\]"):
                registry.load_adapter_from_hub("user/a")


# ── ConcAdptrModel Hub methods ────────────────────────────────────────────────


class TestModelPushToHub:
    def test_calls_upload_folder(self, tmp_path):
        model = _make_registry_model(tmp_path)
        mock_api = MagicMock()
        mock_api.upload_folder.return_value = "https://huggingface.co/user/m"

        with patch("huggingface_hub.HfApi", return_value=mock_api):
            url = model.push_to_hub("user/m", token="tok")

        mock_api.create_repo.assert_called_once()
        mock_api.upload_folder.assert_called_once()
        assert url == "https://huggingface.co/user/m"

    def test_hub_repo_id_saved_in_registry_json(self, tmp_path):
        model = _make_registry_model(tmp_path)
        save_dir = tmp_path / "saved"
        model.save_pretrained(save_dir)

        with open(save_dir / "adapter_registry.json") as f:
            data = json.load(f)

        assert data["b"]["hub_repo_id"] == "user/b-adapter"
        assert data["a"]["hub_repo_id"] is None

    def test_raises_without_huggingface_hub(self, tmp_path):
        model = _make_registry_model(tmp_path)
        with patch.dict("sys.modules", {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="concadptr\\[hub\\]"):
                model.push_to_hub("user/m")


class TestModelFromHub:
    def test_loads_model_and_resolves_adapter_paths(self, tmp_path, monkeypatch, adapter_dir):
        # Set up a fake saved model directory
        save_dir = tmp_path / "hub_model"
        save_dir.mkdir()

        adapter_a_path = adapter_dir("a_hub")
        adapter_b_path = adapter_dir("b_hub")

        registry_data = {
            "a": {"path": str(adapter_a_path), "rank": 16, "alpha": 32,
                  "target_modules": ["q_proj"], "metadata": {}, "hub_repo_id": "user/a-adapter"},
            "b": {"path": str(adapter_b_path), "rank": 16, "alpha": 32,
                  "target_modules": ["q_proj"], "metadata": {}, "hub_repo_id": None},
        }
        with open(save_dir / "adapter_registry.json", "w") as f:
            json.dump(registry_data, f)

        # Write a minimal config YAML
        config = ConcAdptrConfig(
            base_model="fake",
            adapters={"a": str(adapter_a_path), "b": str(adapter_b_path)},
            routing_strategy="soft_merging",
            quantization=None,
        )
        config.save(save_dir / "concadptr_config.yaml")

        # Write dummy router weights
        router = SoftMergingRouter(hidden_size=HIDDEN, num_experts=NUM_EXPERTS, num_layers=2)
        torch.save(router.state_dict(), save_dir / "router.pt")

        def mock_snapshot(repo_id, **kwargs):
            if repo_id == "user/my-concadptr":
                return str(save_dir)
            if repo_id == "user/a-adapter":
                return str(adapter_a_path)
            return str(save_dir)

        def mock_from_config(cls, cfg):
            m = ConcAdptrModel(cfg)
            m.base_model = make_mock_base_model()
            m.router = SoftMergingRouter(hidden_size=HIDDEN, num_experts=NUM_EXPERTS, num_layers=2)
            m.registry._adapters = {
                "a": AdapterInfo(name="a", path=str(adapter_a_path), rank=16),
                "b": AdapterInfo(name="b", path=str(adapter_b_path), rank=16),
            }
            return m

        monkeypatch.setattr(ConcAdptrModel, "from_config", classmethod(mock_from_config))

        with patch("huggingface_hub.snapshot_download", side_effect=mock_snapshot):
            model = ConcAdptrModel.from_hub("user/my-concadptr")

        assert model is not None
