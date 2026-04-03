"""Tests for ConcAdptr core components."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from concadptr.config import ConcAdptrConfig, RouterConfig, RoutingStrategy
from concadptr.router.soft_merging import SoftMergingRouter
from concadptr.router.top_k import TopKRouter
from concadptr.router.xlora import XLoRARouter
from concadptr.adapters import AdapterRegistry


# ── Config Tests ──


class TestConcAdptrConfig:
    def test_default_config(self):
        config = ConcAdptrConfig()
        assert config.routing_strategy == "xlora"
        assert config.quantization == "4bit"
        assert config.freeze_adapters is True

    def test_routing_strategy_sync(self):
        config = ConcAdptrConfig(routing_strategy="top_k")
        assert config.router.strategy == RoutingStrategy.TOP_K

    def test_validation_no_base_model(self):
        config = ConcAdptrConfig(base_model="", adapters={"a": "/tmp/fake"})
        issues = config.validate()
        assert any("base_model" in i for i in issues)

    def test_validation_no_adapters(self):
        config = ConcAdptrConfig(base_model="model", adapters={})
        issues = config.validate()
        assert any("adapter" in i.lower() for i in issues)

    def test_validation_single_adapter_warning(self):
        config = ConcAdptrConfig(base_model="model", adapters={"a": "/tmp/fake"})
        issues = config.validate()
        assert any("WARNING" in i and "2+" in i for i in issues)

    def test_yaml_roundtrip(self):
        config = ConcAdptrConfig(
            base_model="test-model",
            adapters={"a": "/path/a", "b": "/path/b"},
            routing_strategy="soft_merging",
        )
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save(f.name)
            loaded = ConcAdptrConfig.from_yaml(f.name)

        assert loaded.base_model == "test-model"
        assert loaded.adapters == {"a": "/path/a", "b": "/path/b"}
        assert loaded.router.strategy == RoutingStrategy.SOFT_MERGING


# ── Router Tests ──


class TestSoftMergingRouter:
    def test_output_shape(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=2)
        x = torch.randn(2, 10, 64)  # batch=2, seq=10, hidden=64
        weights = router(x, layer_idx=0)
        assert weights.shape == (2, 10, 3)

    def test_weights_sum_to_one(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=0)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_requires_layer_idx(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=2, use_layerwise=True)
        x = torch.randn(1, 5, 64)
        with pytest.raises(ValueError, match="layer_idx"):
            router(x)


class TestTopKRouter:
    def test_sparsity(self):
        router = TopKRouter(hidden_size=64, num_experts=5, num_layers=1, k=2)
        x = torch.randn(1, 10, 64)
        weights = router(x, layer_idx=0)

        # Each token should have exactly k non-zero experts
        nonzero_per_token = (weights > 0).sum(dim=-1)
        assert (nonzero_per_token == 2).all()

    def test_top_k_weights_sum_to_one(self):
        router = TopKRouter(hidden_size=64, num_experts=4, num_layers=1, k=2)
        x = torch.randn(2, 5, 64)
        weights = router(x, layer_idx=0)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestXLoRARouter:
    def test_output_shape(self):
        router = XLoRARouter(
            hidden_size=64,
            num_experts=3,
            num_layers=4,
            classifier_depth=1,
        )
        x = torch.randn(2, 10, 64)
        weights = router(x, layer_idx=0)
        assert weights.shape == (2, 10, 3)

    def test_layer_scalings_shape(self):
        router = XLoRARouter(
            hidden_size=64,
            num_experts=3,
            num_layers=4,
            use_layerwise=True,
        )
        x = torch.randn(1, 5, 64)
        scalings = router.get_layer_scalings(x)
        assert scalings.shape == (1, 5, 4, 3)

    def test_weights_sum_to_one(self):
        router = XLoRARouter(hidden_size=64, num_experts=3, num_layers=2)
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=0)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ── Load Balance Loss Tests ──


class TestLoadBalanceLoss:
    def test_uniform_routing_low_loss(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        # Uniform weights should give loss close to 1.0
        uniform_weights = torch.ones(1, 100, 4) / 4
        loss = router.compute_load_balance_loss(uniform_weights)
        assert loss.item() == pytest.approx(1.0, abs=0.1)

    def test_collapsed_routing_high_loss(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        # All tokens to one expert
        collapsed_weights = torch.zeros(1, 100, 4)
        collapsed_weights[:, :, 0] = 1.0
        loss = router.compute_load_balance_loss(collapsed_weights)
        assert loss.item() > 1.0  # Should be higher than uniform


# ── Routing Stats Tests ──


class TestRoutingStats:
    def test_history_recording(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=1)
        router.enable_history(True)

        x = torch.randn(2, 10, 64)
        router(x, layer_idx=0)
        router(x, layer_idx=0)

        stats = router.get_routing_stats()
        assert "expert_load" in stats
        assert "routing_entropy" in stats
        assert stats["expert_load"].shape == (3,)

    def test_no_history_when_disabled(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=1)
        router.enable_history(False)

        x = torch.randn(2, 10, 64)
        router(x, layer_idx=0)

        stats = router.get_routing_stats()
        assert stats == {}


# ── Adapter Registry Tests ──


class TestAdapterRegistry:
    def _make_adapter_dir(self, tmp_path: Path, name: str, rank: int = 16):
        """Create a fake PEFT adapter directory."""
        adapter_dir = tmp_path / name
        adapter_dir.mkdir()
        config = {
            "base_model_name_or_path": "test-model",
            "r": rank,
            "lora_alpha": rank * 2,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA",
        }
        (adapter_dir / "adapter_config.json").write_text(json.dumps(config))
        # Create a fake weights file
        torch.save({}, adapter_dir / "adapter_model.bin")
        return adapter_dir

    def test_register_adapter(self, tmp_path):
        registry = AdapterRegistry()
        adapter_dir = self._make_adapter_dir(tmp_path, "medical")
        info = registry.register("medical", adapter_dir)
        assert info.name == "medical"
        assert info.rank == 16
        assert registry.num_adapters == 1

    def test_register_duplicate_fails(self, tmp_path):
        registry = AdapterRegistry()
        adapter_dir = self._make_adapter_dir(tmp_path, "medical")
        registry.register("medical", adapter_dir)
        with pytest.raises(ValueError, match="already registered"):
            registry.register("medical", adapter_dir)

    def test_compatibility_check_passes(self, tmp_path):
        registry = AdapterRegistry()
        registry.register("a", self._make_adapter_dir(tmp_path, "a", rank=16))
        registry.register("b", self._make_adapter_dir(tmp_path, "b", rank=16))
        assert registry.validate_compatibility() is True

    def test_compatibility_check_fails_rank_mismatch(self, tmp_path):
        registry = AdapterRegistry()
        registry.register("a", self._make_adapter_dir(tmp_path, "a", rank=16))
        registry.register("b", self._make_adapter_dir(tmp_path, "b", rank=32))
        with pytest.raises(ValueError, match="Rank mismatch"):
            registry.validate_compatibility()

    def test_register_from_dict(self, tmp_path):
        dir_a = self._make_adapter_dir(tmp_path, "a")
        dir_b = self._make_adapter_dir(tmp_path, "b")
        registry = AdapterRegistry()
        registry.register_from_dict({"a": str(dir_a), "b": str(dir_b)})
        assert registry.num_adapters == 2

    def test_summary(self, tmp_path):
        registry = AdapterRegistry()
        registry.register("medical", self._make_adapter_dir(tmp_path, "medical"))
        summary = registry.summary()
        assert "medical" in summary
        assert "rank" in summary.lower() or "Rank" in summary
