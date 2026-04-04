"""Shared fixtures for ConcAdptr tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from concadptr.adapters import AdapterInfo, AdapterRegistry
from concadptr.config import ConcAdptrConfig
from concadptr.router.soft_merging import SoftMergingRouter


# ── Adapter helpers ──


@pytest.fixture
def adapter_dir(tmp_path: Path) -> Callable:
    """Factory: adapter_dir(name, rank, base_model) → Path.

    Writes a minimal PEFT adapter directory to a temp location so
    tests that touch the filesystem don't need to repeat this boilerplate.
    """

    def _make(name: str, rank: int = 16, base_model: str = "test-model") -> Path:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        config = {
            "base_model_name_or_path": base_model,
            "r": rank,
            "lora_alpha": rank * 2,
            "target_modules": ["q_proj", "v_proj"],
            "peft_type": "LORA",
        }
        (d / "adapter_config.json").write_text(json.dumps(config))
        torch.save({}, d / "adapter_model.bin")
        return d

    return _make


# ── Router fixtures ──


@pytest.fixture
def tiny_router() -> SoftMergingRouter:
    """Small SoftMergingRouter usable on CPU in unit tests."""
    return SoftMergingRouter(hidden_size=64, num_experts=2, num_layers=2)


# ── Model mock helpers ──


class _FakeModelOutput:
    """Mimics the HuggingFace CausalLMOutputWithPast returned by the base model."""

    def __init__(self, logits: torch.Tensor, hidden_states: tuple):
        self.logits = logits
        self.hidden_states = hidden_states  # tuple of tensors


def make_mock_base_model(
    batch: int = 1,
    seq: int = 8,
    hidden: int = 64,
    vocab: int = 100,
    num_layers: int = 2,
) -> MagicMock:
    """Return a MagicMock that quacks like an AutoModelForCausalLM.

    The mock:
    - Returns a _FakeModelOutput on __call__ with the right shapes
    - Exposes .config.hidden_size and .config.num_hidden_layers
    - Implements set_adapter() as a no-op
    - Yields a single fake parameter so device detection works
    """
    mock = MagicMock()

    fake_hidden = tuple(
        torch.randn(batch, seq, hidden) for _ in range(num_layers + 1)
    )
    fake_logits = torch.randn(batch, seq, vocab)
    mock.return_value = _FakeModelOutput(logits=fake_logits, hidden_states=fake_hidden)

    mock.config = MagicMock()
    mock.config.hidden_size = hidden
    mock.config.num_hidden_layers = num_layers

    # parameters() must return something iterable with a .device attribute
    fake_param = nn.Parameter(torch.zeros(1))
    mock.parameters.return_value = iter([fake_param])
    mock.named_parameters.return_value = iter([("w", fake_param)])

    mock.set_adapter = MagicMock()

    return mock


@pytest.fixture
def mock_concadptr_model(adapter_dir: Callable) -> "ConcAdptrModel":  # noqa: F821
    """ConcAdptrModel with injected mocks — no HuggingFace downloads needed.

    Injects:
      - A MagicMock base_model with realistic tensor shapes
      - A real SoftMergingRouter so gradient flow is testable
      - Two fake AdapterInfo entries in the registry
    """
    from concadptr.model import ConcAdptrModel

    config = ConcAdptrConfig(
        base_model="fake-model",
        adapters={"a": "/fake/a", "b": "/fake/b"},
        routing_strategy="soft_merging",
        quantization=None,
    )
    model = ConcAdptrModel(config)
    model.base_model = make_mock_base_model()
    model.router = SoftMergingRouter(hidden_size=64, num_experts=2, num_layers=2)
    model.registry._adapters = {
        "a": AdapterInfo(name="a", path="/fake/a", rank=16),
        "b": AdapterInfo(name="b", path="/fake/b", rank=16),
    }
    model._is_loaded = True
    return model
