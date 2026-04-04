"""Tests for ConcAdptrModel: forward pass, save/load, gradient freeze.

Strategy: bypass from_config() entirely by constructing ConcAdptrModel(config)
directly and injecting a MagicMock base_model + real router.  This avoids any
HuggingFace downloads while still exercising real router gradient flow.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from concadptr.adapters import AdapterInfo
from concadptr.config import ConcAdptrConfig
from concadptr.model import ConcAdptrModel
from concadptr.router.soft_merging import SoftMergingRouter

from .conftest import make_mock_base_model

# ── helpers ──

BATCH, SEQ, HIDDEN, VOCAB = 1, 8, 64, 100
NUM_EXPERTS = 2


def _build_model(freeze_adapters: bool = True) -> ConcAdptrModel:
    """Construct a model with injected mocks — no network calls."""
    config = ConcAdptrConfig(
        base_model="fake",
        adapters={"a": "/fake/a", "b": "/fake/b"},
        routing_strategy="soft_merging",
        quantization=None,
        freeze_adapters=freeze_adapters,
    )
    model = ConcAdptrModel(config)
    model.base_model = make_mock_base_model(
        batch=BATCH, seq=SEQ, hidden=HIDDEN, vocab=VOCAB
    )
    model.router = SoftMergingRouter(
        hidden_size=HIDDEN, num_experts=NUM_EXPERTS, num_layers=2
    )
    model.registry._adapters = {
        "a": AdapterInfo(name="a", path="/fake/a", rank=16),
        "b": AdapterInfo(name="b", path="/fake/b", rank=16),
    }
    model._is_loaded = True
    return model


# ── Forward pass ──


class TestConcAdptrModelForward:
    def test_output_keys_present(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids)
        for key in ("loss", "lm_loss", "logits", "routing_weights", "load_balance_loss"):
            assert key in out, f"missing key: {key}"

    def test_logits_shape(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids)
        assert out["logits"].shape == (BATCH, SEQ, VOCAB)

    def test_routing_weights_shape(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids)
        assert out["routing_weights"].shape == (BATCH, SEQ, NUM_EXPERTS)

    def test_no_loss_without_labels(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids)
        assert out["loss"] is None

    def test_loss_computed_with_labels(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        labels = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].ndim == 0  # scalar
        assert out["loss"].item() > 0

    def test_set_adapter_called_for_each_adapter(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        model(input_ids)
        # set_adapter should be called once per adapter per forward pass
        assert model.base_model.set_adapter.call_count == NUM_EXPERTS

    def test_load_balance_loss_is_scalar(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids)
        assert out["load_balance_loss"].ndim == 0

    def test_routing_weights_sum_to_one(self):
        model = _build_model()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids)
        sums = out["routing_weights"].sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ── Gradient freeze ──


class TestGradientFreeze:
    def test_router_params_have_grad(self):
        """Router weights should accumulate gradients during backward."""
        model = _build_model(freeze_adapters=True)
        model.router.train()
        input_ids = torch.zeros(BATCH, SEQ, dtype=torch.long)
        labels = torch.zeros(BATCH, SEQ, dtype=torch.long)
        out = model(input_ids, labels=labels)
        out["loss"].backward()
        for param in model.router.parameters():
            assert param.grad is not None, "Router param missing gradient"
            assert param.grad.abs().sum().item() > 0

    def test_get_trainable_parameters_returns_router_params(self):
        model = _build_model()
        trainable = model.get_trainable_parameters()
        router_params = list(model.router.parameters())
        assert len(trainable) == len(router_params)

    def test_get_num_trainable_params_positive(self):
        model = _build_model()
        n = model.get_num_trainable_params()
        assert n > 0


# ── Save / Load roundtrip ──


class TestSavePretrained:
    def test_creates_required_files(self, tmp_path):
        model = _build_model()
        model.save_pretrained(tmp_path / "saved")
        assert (tmp_path / "saved" / "router.pt").exists()
        assert (tmp_path / "saved" / "concadptr_config.yaml").exists()
        assert (tmp_path / "saved" / "adapter_registry.json").exists()

    def test_adapter_registry_json_contents(self, tmp_path):
        model = _build_model()
        save_dir = tmp_path / "saved"
        model.save_pretrained(save_dir)
        with open(save_dir / "adapter_registry.json") as f:
            registry_data = json.load(f)
        assert "a" in registry_data
        assert "b" in registry_data

    def test_router_weights_roundtrip(self, tmp_path, monkeypatch):
        """Save router weights, reload them into a fresh model, verify they match."""
        model1 = _build_model()

        # Set router weights to a known value
        with torch.no_grad():
            for p in model1.router.parameters():
                p.fill_(0.42)

        save_dir = tmp_path / "saved"
        model1.save_pretrained(save_dir)

        # Patch from_config so load_pretrained doesn't try to download a real model
        def _mock_from_config(cls, cfg):
            m = ConcAdptrModel(cfg)
            m.base_model = make_mock_base_model()
            m.router = SoftMergingRouter(hidden_size=HIDDEN, num_experts=NUM_EXPERTS, num_layers=2)
            m.registry._adapters = {
                "a": AdapterInfo(name="a", path="/fake/a", rank=16),
                "b": AdapterInfo(name="b", path="/fake/b", rank=16),
            }
            return m

        monkeypatch.setattr(ConcAdptrModel, "from_config", classmethod(_mock_from_config))

        model2 = ConcAdptrModel.load_pretrained(save_dir)

        for p1, p2 in zip(model1.router.parameters(), model2.router.parameters()):
            assert torch.allclose(p1, p2), "Router weights don't match after roundtrip"

    def test_save_creates_output_dir(self, tmp_path):
        model = _build_model()
        deep_path = tmp_path / "a" / "b" / "c"
        model.save_pretrained(deep_path)
        assert deep_path.exists()

    def test_repr_contains_key_info(self):
        model = _build_model()
        r = repr(model)
        assert "fake" in r
        assert "soft_merging" in r
