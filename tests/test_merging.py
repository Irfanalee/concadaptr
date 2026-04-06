"""Tests for static adapter merging — Linear, TIES, DARE.

All tests use small synthetic tensors; no real models or network calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import pytest
import torch
from torch import Tensor

from concadptr.adapters import AdapterInfo, AdapterRegistry
from concadptr.merging import DAREMerge, LinearMerge, TIESMerge, merge_adapters
from concadptr.merging.utils import load_adapter_weights, uniform_weights


# ── Helpers ──────────────────────────────────────────────────────────────────


def _write_adapter(path: Path, weights: Dict[str, Tensor], rank: int = 8) -> Path:
    """Write a minimal PEFT adapter directory with given weight tensors."""
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "base_model_name_or_path": "test-model",
        "r": rank,
        "lora_alpha": rank * 2,
        "target_modules": ["q_proj", "v_proj"],
        "peft_type": "LORA",
    }
    (path / "adapter_config.json").write_text(json.dumps(config))
    torch.save(weights, path / "adapter_model.bin")
    return path


def _simple_weights() -> Dict[str, Tensor]:
    """Return small synthetic weight tensors for a 2-module adapter."""
    return {
        "base_model.model.q_proj.lora_A.default.weight": torch.ones(4, 8),
        "base_model.model.q_proj.lora_B.default.weight": torch.zeros(8, 4),
        "base_model.model.v_proj.lora_A.default.weight": torch.full((4, 8), 2.0),
        "base_model.model.v_proj.lora_B.default.weight": torch.full((8, 4), -1.0),
    }


# ── Utils ─────────────────────────────────────────────────────────────────────


class TestUtils:
    def test_uniform_weights_sum_to_one(self):
        w = uniform_weights(4)
        assert abs(sum(w) - 1.0) < 1e-6
        assert all(abs(x - 0.25) < 1e-6 for x in w)

    def test_load_adapter_weights_bin(self, tmp_path):
        weights = {"a": torch.tensor([1.0, 2.0])}
        torch.save(weights, tmp_path / "adapter_model.bin")
        loaded = load_adapter_weights(tmp_path)
        assert "a" in loaded
        assert torch.allclose(loaded["a"], weights["a"])

    def test_load_adapter_weights_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_adapter_weights(tmp_path)


# ── LinearMerge ───────────────────────────────────────────────────────────────


class TestLinearMerge:
    def test_weighted_average_equal_weights(self):
        a = {"w": torch.tensor([2.0, 4.0])}
        b = {"w": torch.tensor([4.0, 8.0])}
        merger = LinearMerge()
        result = merger.merge([a, b], [0.5, 0.5])
        expected = torch.tensor([3.0, 6.0])
        assert torch.allclose(result["w"], expected)

    def test_weighted_average_unequal_weights(self):
        a = {"w": torch.ones(4)}
        b = {"w": torch.zeros(4)}
        merger = LinearMerge()
        result = merger.merge([a, b], [0.75, 0.25])
        assert torch.allclose(result["w"], torch.full((4,), 0.75))

    def test_single_adapter_passthrough(self):
        a = {"w": torch.tensor([1.0, 2.0, 3.0])}
        merger = LinearMerge()
        result = merger.merge([a], [1.0])
        assert torch.allclose(result["w"], a["w"])

    def test_run_creates_valid_peft_dir(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)
        out = tmp_path / "merged"

        merger = LinearMerge()
        result_path = merger.run([path_a, path_b], out)

        assert (result_path / "adapter_config.json").exists()
        assert (result_path / "adapter_model.bin").exists() or (
            result_path / "adapter_model.safetensors"
        ).exists()

    def test_weights_normalized_if_not_summing_to_one(self, tmp_path):
        """Base.run() normalizes weights that don't sum to 1."""
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)

        merger = LinearMerge()
        # Weights [2, 2] should be normalized to [0.5, 0.5]
        result_path = merger.run([path_a, path_b], tmp_path / "out", weights=[2.0, 2.0])
        merged = load_adapter_weights(result_path)

        # With equal normalized weights, merged == original (all same values)
        key = "base_model.model.q_proj.lora_A.default.weight"
        assert torch.allclose(merged[key], w[key])


# ── TIESMerge ────────────────────────────────────────────────────────────────


class TestTIESMerge:
    def test_uniform_adapters_near_equal_output(self):
        """When both adapters are identical, TIES should return ~the same values."""
        t = torch.randn(8, 16)
        a = {"w": t.clone()}
        b = {"w": t.clone()}
        merger = TIESMerge()
        result = merger.merge([a, b], [0.5, 0.5], trim_fraction=0.0)
        assert torch.allclose(result["w"], t, atol=1e-5)

    def test_trim_zeroes_low_magnitude(self):
        """With trim_fraction=0.9, most values should be zeroed out."""
        torch.manual_seed(0)
        t = torch.randn(100)
        a = {"w": t.clone()}
        b = {"w": t.clone()}
        merger = TIESMerge()
        result = merger.merge([a, b], [0.5, 0.5], trim_fraction=0.9)
        # At least 80% of values should be zero (may vary with sign election)
        zero_frac = (result["w"].abs() < 1e-8).float().mean().item()
        assert zero_frac > 0.5  # conservative — trim + sign election removes many

    def test_opposite_sign_adapters_cancel(self):
        """When one adapter is the negative of the other, signs cancel → near zero."""
        t = torch.ones(10)
        a = {"w": t.clone()}
        b = {"w": -t.clone()}
        merger = TIESMerge()
        result = merger.merge([a, b], [0.5, 0.5], trim_fraction=0.0)
        # Elected sign per position should cancel out (sum of signs = 0)
        assert result["w"].abs().max().item() < 1e-5

    def test_output_dtype_preserved(self):
        a = {"w": torch.ones(4, dtype=torch.float16)}
        b = {"w": torch.ones(4, dtype=torch.float16)}
        merger = TIESMerge()
        result = merger.merge([a, b], [0.5, 0.5], trim_fraction=0.0)
        assert result["w"].dtype == torch.float16

    def test_run_produces_valid_peft_dir(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)
        out = tmp_path / "merged_ties"

        merger = TIESMerge()
        result_path = merger.run([path_a, path_b], out, trim_fraction=0.1)

        assert (result_path / "adapter_config.json").exists()


# ── DAREMerge ────────────────────────────────────────────────────────────────


class TestDAREMerge:
    def test_sparsification_reduces_nonzero_count(self):
        """DARE with density=0.5 should drop ~50% of parameters."""
        torch.manual_seed(0)
        t = torch.ones(1000)
        a = {"w": t.clone()}
        b = {"w": t.clone()}
        merger = DAREMerge(use_ties=False)
        # We test by checking the internal sparsification, not via merge() directly.
        # Run merge with density=0.5 and check output is not all ones
        result = merger.merge([a, b], [0.5, 0.5], density=0.5, seed=42)
        # After drop+rescale+average, some values will be 0 (both dropped)
        zero_frac = (result["w"].abs() < 0.1).float().mean().item()
        assert zero_frac > 0.1  # at least some dropped

    def test_rescaling_preserves_expected_magnitude(self):
        """Dropping p fraction and rescaling by 1/density should preserve E[x]."""
        t = torch.ones(10000)
        a = {"w": t.clone()}
        merger = DAREMerge(use_ties=False)
        result = merger.merge([a], [1.0], density=0.5, seed=0)
        # E[output] should be ~1.0 (Bernoulli(0.5) * (1/0.5) = 1.0)
        mean_val = result["w"].mean().item()
        assert abs(mean_val - 1.0) < 0.1  # within 10%

    def test_dare_ties_combined(self):
        """DARE + TIES should produce a valid output without errors."""
        t = torch.randn(50)
        a = {"w": t.clone()}
        b = {"w": t.clone() * 0.5}
        merger = DAREMerge(use_ties=True)
        result = merger.merge([a, b], [0.5, 0.5], density=0.7, seed=1, trim_fraction=0.1)
        assert "w" in result
        assert result["w"].shape == t.shape

    def test_density_out_of_range_raises(self):
        a = {"w": torch.ones(4)}
        merger = DAREMerge()
        with pytest.raises(ValueError, match="density"):
            merger.merge([a], [1.0], density=0.0)

    def test_reproducibility_with_same_seed(self):
        t = torch.randn(100)
        a = {"w": t.clone()}
        b = {"w": t.clone()}
        merger = DAREMerge()
        r1 = merger.merge([a, b], [0.5, 0.5], density=0.6, seed=99)
        r2 = merger.merge([a, b], [0.5, 0.5], density=0.6, seed=99)
        assert torch.allclose(r1["w"], r2["w"])

    def test_different_seeds_differ(self):
        t = torch.randn(100)
        a = {"w": t.clone()}
        b = {"w": t.clone()}
        merger = DAREMerge()
        r1 = merger.merge([a, b], [0.5, 0.5], density=0.6, seed=1)
        r2 = merger.merge([a, b], [0.5, 0.5], density=0.6, seed=2)
        assert not torch.allclose(r1["w"], r2["w"])

    def test_run_produces_valid_peft_dir(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)
        out = tmp_path / "merged_dare"

        merger = DAREMerge()
        result_path = merger.run([path_a, path_b], out, density=0.7, seed=42)

        assert (result_path / "adapter_config.json").exists()


# ── merge_adapters functional API ────────────────────────────────────────────


class TestMergeAdaptersAPI:
    def test_linear_method(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)

        out = merge_adapters(
            adapters={"a": str(path_a), "b": str(path_b)},
            output_path=tmp_path / "out_linear",
            method="linear",
        )
        assert (out / "adapter_config.json").exists()

    def test_ties_method(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)

        out = merge_adapters(
            adapters={"a": str(path_a), "b": str(path_b)},
            output_path=tmp_path / "out_ties",
            method="ties",
            trim_fraction=0.1,
        )
        assert out.exists()

    def test_dare_method(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)

        out = merge_adapters(
            adapters={"a": str(path_a), "b": str(path_b)},
            output_path=tmp_path / "out_dare",
            method="dare",
            density=0.7,
        )
        assert out.exists()

    def test_dare_ties_method(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w)
        path_b = _write_adapter(tmp_path / "b", w)

        out = merge_adapters(
            adapters={"a": str(path_a), "b": str(path_b)},
            output_path=tmp_path / "out_dare_ties",
            method="dare_ties",
            density=0.7,
            trim_fraction=0.1,
        )
        assert out.exists()

    def test_unknown_method_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown merge method"):
            merge_adapters(
                adapters={"a": str(tmp_path)},
                output_path=tmp_path / "out",
                method="foobar",
            )


# ── AdapterRegistry.merge() ──────────────────────────────────────────────────


class TestAdapterRegistryMerge:
    def test_merge_calls_validate_compatibility_first(self, tmp_path):
        """Registry.merge() should raise if adapters have different ranks."""
        w_a = _simple_weights()
        w_b = _simple_weights()

        path_a = _write_adapter(tmp_path / "a", w_a, rank=8)
        path_b = _write_adapter(tmp_path / "b", w_b, rank=16)  # different rank

        registry = AdapterRegistry()
        registry._adapters["a"] = AdapterInfo(name="a", path=str(path_a), rank=8)
        registry._adapters["b"] = AdapterInfo(name="b", path=str(path_b), rank=16)

        with pytest.raises(ValueError, match="Rank mismatch"):
            registry.merge(["a", "b"], tmp_path / "out", method="linear")

    def test_merge_raises_for_missing_adapter(self, tmp_path):
        registry = AdapterRegistry()
        with pytest.raises(KeyError):
            registry.merge(["nonexistent"], tmp_path / "out")

    def test_merge_linear_produces_output(self, tmp_path):
        w = _simple_weights()
        path_a = _write_adapter(tmp_path / "a", w, rank=8)
        path_b = _write_adapter(tmp_path / "b", w, rank=8)

        registry = AdapterRegistry()
        registry._adapters["a"] = AdapterInfo(
            name="a", path=str(path_a), rank=8,
            target_modules=["q_proj", "v_proj"]
        )
        registry._adapters["b"] = AdapterInfo(
            name="b", path=str(path_b), rank=8,
            target_modules=["q_proj", "v_proj"]
        )

        out = registry.merge(["a", "b"], tmp_path / "merged", method="linear")
        assert out.exists()
        assert (out / "adapter_config.json").exists()
