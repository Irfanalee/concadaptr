"""Tests for ConcAdptrTrainer: gradient accumulation and step counting.

Strategy: replace the real model with a lightweight fake that returns
pre-built loss tensors.  This avoids loading any HuggingFace model while
still exercising the training-loop logic (accumulation, optimizer steps,
global_step increments).
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from concadptr.config import ConcAdptrConfig
from concadptr.trainer import ConcAdptrTrainer


# ── Fake dataset ──


class _TextDataset(Dataset):
    """Tiny dataset of string examples."""

    def __init__(self, n: int = 8):
        self.data = [f"example {i}" for i in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ── Fake tokenizer / model ──


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, texts, max_length=16, truncation=True, padding=None, return_tensors=None):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones(n, max_length, dtype=torch.long),
            "attention_mask": torch.ones(n, max_length, dtype=torch.long),
        }


class _FakeRouter(nn.Module):
    """Single-parameter router so AdamW has something to update."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))

    def train(self, mode=True):
        return super().train(mode)

    def eval(self):
        return super().eval()


class _FakeModel:
    """Minimal stand-in for ConcAdptrModel."""

    def __init__(self):
        self.router = _FakeRouter()
        self.tokenizer = _FakeTokenizer()
        self.config = ConcAdptrConfig(routing_strategy="soft_merging")
        self.registry = MagicMock()
        self.registry.names = ["a", "b"]

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        return list(self.router.parameters())

    def get_num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.router.parameters())

    def __call__(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Always return a non-None loss so the trainer loop doesn't skip
        loss = (self.router.w ** 2).sum()  # depends on router param → real gradient
        return {
            "loss": loss,
            "lm_loss": loss,
            "logits": None,
            "routing_weights": torch.ones(1, 1, 2) / 2,
            "load_balance_loss": torch.tensor(0.01),
        }

    def save_pretrained(self, path):
        import pathlib
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# ── Tests ──


class TestConcAdptrTrainer:
    def _make_trainer(
        self,
        tmp_path,
        num_samples: int = 8,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 2,
        num_epochs: int = 1,
    ) -> ConcAdptrTrainer:
        dataset = _TextDataset(n=num_samples)
        return ConcAdptrTrainer(
            model=_FakeModel(),
            train_dataset=dataset,
            output_dir=str(tmp_path / "output"),
            num_epochs=num_epochs,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            fp16=False,
            logging_steps=1000,
            eval_steps=1000,
            save_steps=1000,
        )

    def test_global_step_increments_by_accumulation(self, tmp_path):
        """global_step should equal floor(num_batches / accum_steps)."""
        # 8 samples / batch_size 2 = 4 batches per epoch
        # floor(4 / 2) = 2 optimizer steps
        trainer = self._make_trainer(
            tmp_path, num_samples=8, batch_size=2, gradient_accumulation_steps=2
        )
        trainer.train()
        assert trainer.global_step == 2

    def test_global_step_single_accumulation(self, tmp_path):
        """With accum_steps=1, every batch triggers an optimizer step."""
        trainer = self._make_trainer(
            tmp_path, num_samples=4, batch_size=1, gradient_accumulation_steps=1
        )
        trainer.train()
        assert trainer.global_step == 4

    def test_train_returns_dict_with_expected_keys(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        result = trainer.train()
        for key in ("total_steps", "training_time_seconds", "best_eval_loss", "final_loss"):
            assert key in result, f"missing key: {key}"

    def test_total_steps_matches_global_step(self, tmp_path):
        trainer = self._make_trainer(tmp_path)
        result = trainer.train()
        assert result["total_steps"] == trainer.global_step

    def test_router_weights_change_after_training(self, tmp_path):
        """The optimizer must actually update the router parameter."""
        trainer = self._make_trainer(tmp_path, num_samples=4, batch_size=2, gradient_accumulation_steps=1)
        w_before = trainer.model.router.w.item()
        trainer.train()
        w_after = trainer.model.router.w.item()
        assert w_before != pytest.approx(w_after), "Router weights unchanged — optimizer did not step"

    def test_multi_epoch_accumulates_steps(self, tmp_path):
        """global_step accumulates across epochs."""
        # 4 samples, batch=2 → 2 batches/epoch, accum=1 → 2 steps/epoch
        # 3 epochs → 6 total steps
        trainer = self._make_trainer(
            tmp_path, num_samples=4, batch_size=2, gradient_accumulation_steps=1, num_epochs=3
        )
        trainer.train()
        assert trainer.global_step == 6
