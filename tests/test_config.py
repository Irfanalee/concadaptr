"""Tests for ConcAdptrConfig and related config classes."""

from __future__ import annotations

import tempfile

import pytest

from concadptr.config import ConcAdptrConfig, RouterConfig, RoutingStrategy


class TestConcAdptrConfig:
    def test_default_config(self):
        config = ConcAdptrConfig()
        assert config.routing_strategy == "xlora"
        assert config.quantization == "4bit"
        assert config.freeze_adapters is True

    def test_routing_strategy_sync(self):
        config = ConcAdptrConfig(routing_strategy="top_k")
        assert config.router.strategy == RoutingStrategy.TOP_K

    def test_all_strategies_sync(self):
        for strategy in ("soft_merging", "top_k", "xlora"):
            config = ConcAdptrConfig(routing_strategy=strategy)
            assert config.router.strategy == RoutingStrategy(strategy)

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

    def test_validation_invalid_quantization(self):
        config = ConcAdptrConfig(
            base_model="model",
            adapters={"a": "/tmp/a", "b": "/tmp/b"},
            quantization="3bit",
        )
        issues = config.validate()
        assert any("ERROR" in i and "quantization" in i for i in issues)

    def test_validation_top_k_exceeds_adapters(self):
        config = ConcAdptrConfig(
            base_model="model",
            adapters={"a": "/tmp/a", "b": "/tmp/b"},
            routing_strategy="top_k",
        )
        config.router.num_experts_per_token = 5  # more than 2 adapters
        issues = config.validate()
        assert any("num_experts_per_token" in i for i in issues)

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

    def test_yaml_roundtrip_preserves_router_params(self):
        config = ConcAdptrConfig(
            base_model="model",
            adapters={"a": "/a", "b": "/b"},
            routing_strategy="xlora",
        )
        config.router.load_balance_weight = 0.05
        config.router.softmax_temperature = 0.5

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            config.save(f.name)
            loaded = ConcAdptrConfig.from_yaml(f.name)

        assert loaded.router.load_balance_weight == pytest.approx(0.05)
        assert loaded.router.softmax_temperature == pytest.approx(0.5)

    def test_freeze_flags_default(self):
        config = ConcAdptrConfig()
        assert config.freeze_adapters is True
        assert config.freeze_base_model is True
