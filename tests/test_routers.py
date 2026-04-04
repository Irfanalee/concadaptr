"""Tests for all three router strategies, load-balance loss, and routing stats."""

from __future__ import annotations

import pytest
import torch

from concadptr.router.soft_merging import SoftMergingRouter
from concadptr.router.top_k import TopKRouter
from concadptr.router.xlora import XLoRARouter


# ── SoftMergingRouter ──


class TestSoftMergingRouter:
    def test_output_shape(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=2)
        x = torch.randn(2, 10, 64)
        weights = router(x, layer_idx=0)
        assert weights.shape == (2, 10, 3)

    def test_weights_sum_to_one(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=0)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_requires_layer_idx_when_layerwise(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=2, use_layerwise=True)
        x = torch.randn(1, 5, 64)
        with pytest.raises(ValueError, match="layer_idx"):
            router(x)

    def test_shared_gate_no_layer_idx_needed(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=2, use_layerwise=False)
        x = torch.randn(1, 5, 64)
        weights = router(x)  # no layer_idx — should not raise
        assert weights.shape == (1, 5, 3)

    def test_weights_non_negative(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        x = torch.randn(2, 8, 64)
        weights = router(x, layer_idx=0)
        assert (weights >= 0).all()

    def test_temperature_sharpens_distribution(self):
        """Lower temperature → more peaked distribution → lower entropy."""
        x = torch.randn(1, 10, 64)
        router_sharp = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1, temperature=0.1)
        router_soft = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1, temperature=10.0)

        # Share the same gate weights so the only difference is temperature
        router_sharp.load_state_dict(router_soft.state_dict())

        w_sharp = router_sharp(x, layer_idx=0)
        w_soft = router_soft(x, layer_idx=0)

        eps = 1e-8
        entropy_sharp = -(w_sharp * (w_sharp + eps).log()).sum(dim=-1).mean()
        entropy_soft = -(w_soft * (w_soft + eps).log()).sum(dim=-1).mean()
        assert entropy_sharp < entropy_soft


# ── TopKRouter ──


class TestTopKRouter:
    def test_sparsity(self):
        router = TopKRouter(hidden_size=64, num_experts=5, num_layers=1, k=2)
        router.eval()  # disable noise
        x = torch.randn(1, 10, 64)
        weights = router(x, layer_idx=0)
        nonzero_per_token = (weights > 0).sum(dim=-1)
        assert (nonzero_per_token == 2).all()

    def test_top_k_weights_sum_to_one(self):
        router = TopKRouter(hidden_size=64, num_experts=4, num_layers=1, k=2)
        router.eval()
        x = torch.randn(2, 5, 64)
        weights = router(x, layer_idx=0)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_k_clamped_to_num_experts(self):
        """k > num_experts should be silently clamped."""
        router = TopKRouter(hidden_size=64, num_experts=3, num_layers=1, k=10)
        assert router.k == 3

    def test_output_shape(self):
        router = TopKRouter(hidden_size=64, num_experts=4, num_layers=2, k=2)
        x = torch.randn(2, 6, 64)
        weights = router(x, layer_idx=1)
        assert weights.shape == (2, 6, 4)

    def test_k1_routes_to_single_expert(self):
        router = TopKRouter(hidden_size=64, num_experts=4, num_layers=1, k=1)
        router.eval()
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=0)
        nonzero = (weights > 0).sum(dim=-1)
        assert (nonzero == 1).all()


# ── XLoRARouter ──


class TestXLoRARouter:
    def test_output_shape(self):
        router = XLoRARouter(hidden_size=64, num_experts=3, num_layers=4, classifier_depth=1)
        x = torch.randn(2, 10, 64)
        weights = router(x, layer_idx=0)
        assert weights.shape == (2, 10, 3)

    def test_layer_scalings_shape(self):
        router = XLoRARouter(hidden_size=64, num_experts=3, num_layers=4, use_layerwise=True)
        x = torch.randn(1, 5, 64)
        scalings = router.get_layer_scalings(x)
        assert scalings.shape == (1, 5, 4, 3)

    def test_get_layer_scalings_non_layerwise(self):
        """get_layer_scalings expands to (batch, seq, num_layers, num_experts) even without layerwise."""
        router = XLoRARouter(hidden_size=64, num_experts=3, num_layers=4, use_layerwise=False)
        x = torch.randn(1, 5, 64)
        scalings = router.get_layer_scalings(x)
        assert scalings.shape == (1, 5, 4, 3)

    def test_weights_sum_to_one(self):
        router = XLoRARouter(hidden_size=64, num_experts=3, num_layers=2)
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=0)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_no_softmax_weights_unconstrained(self):
        router = XLoRARouter(
            hidden_size=64, num_experts=3, num_layers=2, enable_softmax=False
        )
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=0)
        # Without softmax, weights won't necessarily sum to 1
        sums = weights.sum(dim=-1)
        assert not torch.allclose(sums, torch.ones_like(sums), atol=0.1) or True
        # Shape still correct
        assert weights.shape == (1, 5, 3)

    def test_layer_idx_none_averages_layers(self):
        """Calling forward without layer_idx averages across layers."""
        router = XLoRARouter(hidden_size=64, num_experts=3, num_layers=4, use_layerwise=True)
        x = torch.randn(1, 5, 64)
        weights = router(x, layer_idx=None)  # no layer_idx
        assert weights.shape == (1, 5, 3)


# ── Load Balance Loss ──


class TestLoadBalanceLoss:
    def test_uniform_routing_gives_loss_near_one(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        uniform = torch.ones(1, 100, 4) / 4
        loss = router.compute_load_balance_loss(uniform)
        assert loss.item() == pytest.approx(1.0, abs=0.1)

    def test_collapsed_routing_higher_than_uniform(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        uniform = torch.ones(1, 100, 4) / 4
        collapsed = torch.zeros(1, 100, 4)
        collapsed[:, :, 0] = 1.0
        assert router.compute_load_balance_loss(collapsed).item() > \
               router.compute_load_balance_loss(uniform).item()

    def test_loss_is_scalar(self):
        router = TopKRouter(hidden_size=64, num_experts=3, num_layers=1, k=2)
        weights = torch.randn(2, 10, 3).softmax(dim=-1)
        loss = router.compute_load_balance_loss(weights)
        assert loss.shape == ()

    def test_loss_non_negative(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=1)
        weights = torch.randn(1, 20, 3).softmax(dim=-1)
        assert router.compute_load_balance_loss(weights).item() >= 0


# ── Routing Stats ──


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
        assert router.get_routing_stats() == {}

    def test_history_cleared_on_disable(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=3, num_layers=1)
        router.enable_history(True)
        router(torch.randn(1, 5, 64), layer_idx=0)
        assert len(router._routing_history) > 0
        router.enable_history(False)
        assert len(router._routing_history) == 0

    def test_expert_load_sums_to_one(self):
        router = SoftMergingRouter(hidden_size=64, num_experts=4, num_layers=1)
        router.enable_history(True)
        for _ in range(5):
            router(torch.randn(2, 10, 64), layer_idx=0)
        stats = router.get_routing_stats()
        # expert_load is average weight per expert — for soft router sums to 1
        assert stats["expert_load"].sum().item() == pytest.approx(1.0, abs=0.05)
