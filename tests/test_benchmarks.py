"""Tests for the ConcAdptr benchmarking suite.

All tests use mocks — no real models, no HuggingFace downloads.

Strategy:
- TestMetrics: pure unit tests, no mocking needed.
- TestMMLUTask / TestHellaSwagTask: patch datasets.load_dataset + _score_choices.
- TestGenerationTask: patch datasets.load_dataset + model generate methods.
- TestBenchmarkRunner: mock task registry to avoid real evaluation.
- TestForgettingCheck: same mock approach, verify delta computation.
"""

from __future__ import annotations

import sys
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from concadptr.adapters import AdapterInfo
from concadptr.benchmarks.config import BenchmarkConfig, BenchmarkResult
from concadptr.benchmarks.metrics import accuracy, f1_score
from concadptr.benchmarks.runner import BenchmarkRunner
from concadptr.benchmarks.tasks import (
    GenerationTask,
    HellaSwagTask,
    MMLUTask,
    _TASK_REGISTRY,
    _score_choices,
)
from concadptr.config import ConcAdptrConfig
from concadptr.model import ConcAdptrModel
from concadptr.router.soft_merging import SoftMergingRouter

from .conftest import make_mock_base_model

# ── Shared helpers ────────────────────────────────────────────────────────────

HIDDEN, VOCAB = 64, 100
NUM_EXPERTS = 2


def _make_model() -> ConcAdptrModel:
    """Build a ConcAdptrModel with injected mocks — no network calls."""
    config = ConcAdptrConfig(
        base_model="fake",
        adapters={"a": "/fake/a", "b": "/fake/b"},
        routing_strategy="soft_merging",
        quantization=None,
    )
    model = ConcAdptrModel(config)
    model.base_model = make_mock_base_model(hidden=HIDDEN, vocab=VOCAB)
    model.router = SoftMergingRouter(hidden_size=HIDDEN, num_experts=NUM_EXPERTS, num_layers=2)
    model.registry._adapters = {
        "a": AdapterInfo(name="a", path="/fake/a", rank=16),
        "b": AdapterInfo(name="b", path="/fake/b", rank=16),
    }
    model._is_loaded = True
    return model


def _make_result(task: str = "mmlu", adapter_name: str = "concadptr_routed") -> BenchmarkResult:
    return BenchmarkResult(
        task=task,
        adapter_name=adapter_name,
        metrics={"accuracy": 0.7},
        num_samples=10,
        elapsed_seconds=0.5,
    )


def _fake_mmlu_dataset(n: int = 3) -> list:
    """Return n fake MMLU-style examples. All correct answers are index 2."""
    return [
        {
            "question": f"Question {i}?",
            "choices": ["A", "B", "C", "D"],
            "answer": 2,
            "subject": "math",
        }
        for i in range(n)
    ]


def _fake_hellaswag_dataset(n: int = 3) -> list:
    """Return n fake HellaSwag-style examples. All correct labels are 1."""
    return [
        {
            "ctx": f"Context sentence {i}.",
            "endings": ["end_0", "end_1", "end_2", "end_3"],
            "label": "1",
        }
        for i in range(n)
    ]


def _fake_generation_dataset(n: int = 3) -> list:
    return [
        {"input": f"Input {i}", "output": f"Reference {i}"}
        for i in range(n)
    ]


# ── TestMetrics ───────────────────────────────────────────────────────────────


class TestAccuracy:
    def test_perfect(self):
        assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0

    def test_all_wrong(self):
        assert accuracy([1, 2, 3], [0, 1, 2]) == 0.0

    def test_partial(self):
        assert accuracy([0, 1, 0], [0, 1, 2]) == pytest.approx(2 / 3)

    def test_empty(self):
        assert accuracy([], []) == 0.0

    def test_single_correct(self):
        assert accuracy([3], [3]) == 1.0


class TestF1Score:
    def test_perfect_binary(self):
        assert f1_score([1, 1, 0], [1, 1, 0], average="binary") == pytest.approx(1.0)

    def test_all_wrong_binary(self):
        assert f1_score([0, 0, 0], [1, 1, 1], average="binary") == pytest.approx(0.0)

    def test_macro_uniform(self):
        # Perfect predictions on 3 classes → F1 = 1.0
        preds = [0, 1, 2, 0, 1, 2]
        labels = [0, 1, 2, 0, 1, 2]
        assert f1_score(preds, labels, average="macro") == pytest.approx(1.0)

    def test_empty(self):
        assert f1_score([], []) == 0.0

    def test_macro_partial(self):
        # Predicts all class 0 on a balanced 2-class problem
        preds = [0, 0, 0, 0]
        labels = [0, 0, 1, 1]
        score = f1_score(preds, labels, average="macro")
        assert 0.0 <= score <= 1.0


class TestMetricsImportErrors:
    def test_bleu_raises_without_evaluate(self, monkeypatch):
        """bleu() raises ImportError with install hint when evaluate is missing."""
        import concadptr.benchmarks.metrics as m

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "evaluate":
                raise ImportError("No module named 'evaluate'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        with pytest.raises(ImportError, match="concadptr\\[benchmarks\\]"):
            m.bleu(["hello"], ["hello"])

    def test_rouge_raises_without_evaluate(self, monkeypatch):
        """rouge() raises ImportError with install hint when evaluate is missing."""
        import concadptr.benchmarks.metrics as m

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def mock_import(name, *args, **kwargs):
            if name == "evaluate":
                raise ImportError("No module named 'evaluate'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", mock_import)
        with pytest.raises(ImportError, match="concadptr\\[benchmarks\\]"):
            m.rouge(["hello"], ["hello"])


# ── TestScoreChoices ──────────────────────────────────────────────────────────


class TestScoreChoices:
    """Tests for _score_choices log-prob scoring logic."""

    def _make_tokenizer(self, prefix_len: int = 2, choice_lens: List[int] = None):
        """Return a mock tokenizer where call N returns tokens of a known length."""
        if choice_lens is None:
            choice_lens = [3, 1]

        call_count = [0]

        def side_effect(text, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == 0:
                # prefix call
                length = prefix_len
            else:
                # full text call for choice (idx-1)
                choice_idx = idx - 1
                length = prefix_len + choice_lens[choice_idx % len(choice_lens)]
            ids = torch.zeros(1, length, dtype=torch.long)
            return {"input_ids": ids}

        tok = MagicMock()
        tok.side_effect = side_effect
        return tok

    def test_returns_highest_scoring_choice(self):
        """Choice 0 should win when its logits strongly favor the expected tokens."""
        model = _make_model()

        # prefix_len=2, choice 0 has 3 tokens, choice 1 has 1 token
        # choice 0 tokens: [0, 0, 0] — logits[0, :, 0] = 100 → log_softmax ≈ 0
        # choice 1 tokens: [0] — uniform logits → log_softmax ≈ -log(vocab)
        # choice 0 sum (≈0*3=0) > choice 1 sum (≈-log(100)*1 ≈ -4.6)
        call_count = [0]
        vocab = VOCAB

        def mock_get_logits(m, input_ids, adapter_name):
            seq_len = input_ids.shape[1]
            logits = torch.zeros(1, seq_len, vocab)
            if call_count[0] == 0:
                # Choice 0: make token 0 very likely at all answer positions
                logits[0, :, 0] = 100.0
            # Choice 1: uniform logits (low score)
            call_count[0] += 1
            return logits

        tok = self._make_tokenizer(prefix_len=2, choice_lens=[3, 1])
        device = torch.device("cpu")

        with patch("concadptr.benchmarks.tasks._get_logits", side_effect=mock_get_logits):
            result = _score_choices(model, tok, "prefix ", ["choice_a", "choice_b"], device, "base")

        assert result == 0

    def test_returns_second_choice_when_logits_favor_it(self):
        """Choice 1 should win when its logits give a higher sum."""
        model = _make_model()
        call_count = [0]
        vocab = VOCAB

        def mock_get_logits(m, input_ids, adapter_name):
            seq_len = input_ids.shape[1]
            logits = torch.zeros(1, seq_len, vocab)
            if call_count[0] == 1:
                # Choice 1: make token 0 very likely
                logits[0, :, 0] = 100.0
            call_count[0] += 1
            return logits

        tok = self._make_tokenizer(prefix_len=2, choice_lens=[1, 3])
        device = torch.device("cpu")

        with patch("concadptr.benchmarks.tasks._get_logits", side_effect=mock_get_logits):
            result = _score_choices(model, tok, "prefix ", ["choice_a", "choice_b"], device, "base")

        assert result == 1

    def test_handles_single_choice(self):
        """Single choice should always return index 0."""
        model = _make_model()

        def mock_get_logits(m, input_ids, adapter_name):
            return torch.zeros(1, input_ids.shape[1], VOCAB)

        tok = self._make_tokenizer(prefix_len=2, choice_lens=[2])
        device = torch.device("cpu")

        with patch("concadptr.benchmarks.tasks._get_logits", side_effect=mock_get_logits):
            result = _score_choices(model, tok, "prefix ", ["only_choice"], device, "base")

        assert result == 0


# ── TestMMLUTask ──────────────────────────────────────────────────────────────


class TestMMLUTask:
    def _run_with_mocked_dataset(self, mocker, preds: List[int], num_samples: int = None):
        """Run MMLUTask with mocked dataset and _score_choices returning given preds."""
        fake_data = _fake_mmlu_dataset(len(preds))
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))
        mock_ds.__len__ = MagicMock(return_value=len(fake_data))
        mock_ds.filter.return_value = mock_ds
        mock_ds.select.return_value = mock_ds

        mocker.patch("datasets.load_dataset", return_value=mock_ds)

        pred_iter = iter(preds)
        mocker.patch(
            "concadptr.benchmarks.tasks._score_choices",
            side_effect=lambda *a, **kw: next(pred_iter),
        )

        model = _make_model()
        config = BenchmarkConfig(num_samples=num_samples)
        return MMLUTask().evaluate(model, config)

    def test_result_has_accuracy_key(self, mocker):
        result = self._run_with_mocked_dataset(mocker, [2, 2, 2])
        assert "accuracy" in result.metrics

    def test_perfect_score(self, mocker):
        """All predictions match labels (all 2) → accuracy = 1.0."""
        result = self._run_with_mocked_dataset(mocker, [2, 2, 2])
        assert result.metrics["accuracy"] == pytest.approx(1.0)

    def test_zero_score(self, mocker):
        """No predictions match labels → accuracy = 0.0."""
        result = self._run_with_mocked_dataset(mocker, [0, 0, 0])
        assert result.metrics["accuracy"] == pytest.approx(0.0)

    def test_result_fields(self, mocker):
        result = self._run_with_mocked_dataset(mocker, [2, 0, 2])
        assert result.task == "mmlu"
        assert result.adapter_name == "concadptr_routed"
        assert result.num_samples == 3
        assert result.elapsed_seconds >= 0

    def test_adapter_name_passed_through(self, mocker):
        """adapter_name is forwarded to _score_choices."""
        fake_data = _fake_mmlu_dataset(1)
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))
        mock_ds.filter.return_value = mock_ds
        mock_ds.select.return_value = mock_ds
        mocker.patch("datasets.load_dataset", return_value=mock_ds)

        captured = []

        def mock_score(model, tokenizer, prefix, choices, device, adapter_name, **kw):
            captured.append(adapter_name)
            return 2

        mocker.patch("concadptr.benchmarks.tasks._score_choices", side_effect=mock_score)

        model = _make_model()
        MMLUTask().evaluate(model, BenchmarkConfig(), adapter_name="my_adapter")
        assert captured[0] == "my_adapter"


# ── TestHellaSwagTask ─────────────────────────────────────────────────────────


class TestHellaSwagTask:
    def _run_with_mocked_dataset(self, mocker, preds: List[int]):
        fake_data = _fake_hellaswag_dataset(len(preds))
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))
        mock_ds.select.return_value = mock_ds

        mocker.patch("datasets.load_dataset", return_value=mock_ds)

        pred_iter = iter(preds)
        mocker.patch(
            "concadptr.benchmarks.tasks._score_choices",
            side_effect=lambda *a, **kw: next(pred_iter),
        )

        model = _make_model()
        return HellaSwagTask().evaluate(model, BenchmarkConfig())

    def test_result_has_accuracy_key(self, mocker):
        result = self._run_with_mocked_dataset(mocker, [1, 1, 1])
        assert "accuracy" in result.metrics

    def test_perfect_score(self, mocker):
        """All preds match label=1 → accuracy = 1.0."""
        result = self._run_with_mocked_dataset(mocker, [1, 1, 1])
        assert result.metrics["accuracy"] == pytest.approx(1.0)

    def test_result_task_name(self, mocker):
        result = self._run_with_mocked_dataset(mocker, [1])
        assert result.task == "hellaswag"

    def test_num_samples_in_result(self, mocker):
        result = self._run_with_mocked_dataset(mocker, [1, 0, 1])
        assert result.num_samples == 3


# ── TestGenerationTask ────────────────────────────────────────────────────────


class TestGenerationTask:
    def _make_gen_model(self) -> ConcAdptrModel:
        model = _make_model()
        # tokenizer.decode returns a plain string
        model.tokenizer = MagicMock()
        model.tokenizer.return_value = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
        model.tokenizer.decode.return_value = "generated text"
        return model

    def _run(self, mocker, preds: List[str] = None, n: int = 2):
        fake_data = _fake_generation_dataset(n)
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter(fake_data))
        mock_ds.select.return_value = mock_ds
        mocker.patch("datasets.load_dataset", return_value=mock_ds)
        mocker.patch("concadptr.benchmarks.tasks.bleu", return_value=34.5)
        mocker.patch("concadptr.benchmarks.tasks.rouge", return_value={"rouge1": 0.5, "rouge2": 0.3, "rougeL": 0.4})

        model = self._make_gen_model()
        config = BenchmarkConfig(
            generation_dataset="fake/dataset",
            generation_metrics=["bleu", "rouge"],
        )
        return GenerationTask().evaluate(model, config)

    def test_result_contains_bleu_and_rouge(self, mocker):
        result = self._run(mocker)
        assert "bleu" in result.metrics
        assert "rouge1" in result.metrics
        assert "rougeL" in result.metrics

    def test_raises_without_generation_dataset(self):
        model = _make_model()
        with pytest.raises(ValueError, match="generation_dataset"):
            GenerationTask().evaluate(model, BenchmarkConfig())

    def test_task_name_is_generation(self, mocker):
        result = self._run(mocker)
        assert result.task == "generation"

    def test_num_samples_matches_dataset(self, mocker):
        result = self._run(mocker, n=2)
        assert result.num_samples == 2


# ── TestBenchmarkRunner ───────────────────────────────────────────────────────


class TestBenchmarkRunner:
    def _patched_registry(self, mocker, task_name: str = "mmlu", acc: float = 0.7):
        """Patch task registry so 'mmlu' returns a fake result instantly."""
        mock_result = _make_result(task=task_name)
        mock_result.metrics = {"accuracy": acc}
        mock_task_cls = MagicMock()
        mock_task_cls.return_value.evaluate.return_value = mock_result
        mocker.patch.dict(
            "concadptr.benchmarks.tasks._TASK_REGISTRY",
            {task_name: mock_task_cls},
            clear=True,
        )
        return mock_task_cls

    def test_run_returns_list_of_results(self, mocker):
        self._patched_registry(mocker)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig(tasks=["mmlu"]))
        results = runner.run()
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].task == "mmlu"

    def test_run_skips_unknown_tasks(self, mocker):
        self._patched_registry(mocker, "mmlu")
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig(tasks=["mmlu", "nonexistent"]))
        results = runner.run()
        assert len(results) == 1

    def test_run_tasks_override(self, mocker):
        """run(tasks=[...]) overrides config.tasks."""
        self._patched_registry(mocker)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig(tasks=["hellaswag"]))
        results = runner.run(tasks=["mmlu"])
        assert results[0].task == "mmlu"

    def test_compare_has_base_and_routed_keys(self, mocker):
        self._patched_registry(mocker)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig(tasks=["mmlu"]))
        comparison = runner.compare(adapter_names=[])
        assert "base" in comparison
        assert "concadptr_routed" in comparison

    def test_compare_includes_each_adapter(self, mocker):
        self._patched_registry(mocker)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig(tasks=["mmlu"]))
        comparison = runner.compare(adapter_names=["a", "b"])
        assert "a" in comparison
        assert "b" in comparison

    def test_compare_passes_correct_adapter_names(self, mocker):
        """Each variant passes the right adapter_name to task.evaluate()."""
        captured: List[str] = []

        def mock_evaluate(m, config, adapter_name="concadptr_routed"):
            captured.append(adapter_name)
            return _make_result(adapter_name=adapter_name)

        mock_task_cls = MagicMock()
        mock_task_cls.return_value.evaluate.side_effect = mock_evaluate
        mocker.patch.dict(
            "concadptr.benchmarks.tasks._TASK_REGISTRY",
            {"mmlu": mock_task_cls},
            clear=True,
        )

        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig(tasks=["mmlu"]))
        runner.compare(adapter_names=["a"])

        assert "base" in captured
        assert "a" in captured
        assert "concadptr_routed" in captured

    def test_run_saves_json_when_output_dir_set(self, mocker, tmp_path):
        self._patched_registry(mocker)
        model = _make_model()
        runner = BenchmarkRunner(
            model, BenchmarkConfig(tasks=["mmlu"], output_dir=str(tmp_path))
        )
        runner.run()
        assert (tmp_path / "benchmark_results.json").exists()

    def test_compare_saves_compare_json(self, mocker, tmp_path):
        self._patched_registry(mocker)
        model = _make_model()
        runner = BenchmarkRunner(
            model, BenchmarkConfig(tasks=["mmlu"], output_dir=str(tmp_path))
        )
        runner.compare(adapter_names=[])
        assert (tmp_path / "compare_results.json").exists()


# ── TestForgettingCheck ───────────────────────────────────────────────────────


class TestForgettingCheck:
    def _patch_both_tasks(self, mocker, base_acc: float, routed_acc: float):
        """Patch mmlu + hellaswag to return specified accuracies."""
        results_by_adapter: Dict[str, float] = {
            "base": base_acc,
            "concadptr_routed": routed_acc,
        }

        def make_evaluate(task_name: str):
            def mock_evaluate(m, config, adapter_name="concadptr_routed"):
                acc = results_by_adapter.get(adapter_name, 0.5)
                result = _make_result(task=task_name)
                result.metrics = {"accuracy": acc}
                return result
            return mock_evaluate

        patched: Dict[str, MagicMock] = {}
        for task_name in ["mmlu", "hellaswag"]:
            mock_cls = MagicMock()
            mock_cls.return_value.evaluate.side_effect = make_evaluate(task_name)
            patched[task_name] = mock_cls

        mocker.patch.dict(
            "concadptr.benchmarks.tasks._TASK_REGISTRY",
            patched,
        )

    def test_returns_mmlu_and_hellaswag_keys(self, mocker):
        self._patch_both_tasks(mocker, 0.7, 0.68)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig())
        result = runner.forgetting_check()
        assert "mmlu" in result
        assert "hellaswag" in result

    def test_delta_is_routed_minus_base(self, mocker):
        self._patch_both_tasks(mocker, base_acc=0.70, routed_acc=0.68)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig())
        result = runner.forgetting_check()
        # delta = routed - base = 0.68 - 0.70 = -0.02
        assert result["mmlu"]["delta"] == pytest.approx(-0.02, abs=1e-5)

    def test_no_forgetting_when_equal(self, mocker):
        self._patch_both_tasks(mocker, base_acc=0.75, routed_acc=0.75)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig())
        result = runner.forgetting_check()
        assert result["mmlu"]["delta"] == pytest.approx(0.0, abs=1e-5)

    def test_positive_delta_when_routed_is_better(self, mocker):
        self._patch_both_tasks(mocker, base_acc=0.65, routed_acc=0.70)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig())
        result = runner.forgetting_check()
        assert result["mmlu"]["delta"] > 0

    def test_result_has_base_routed_delta_keys(self, mocker):
        self._patch_both_tasks(mocker, 0.7, 0.69)
        model = _make_model()
        runner = BenchmarkRunner(model, BenchmarkConfig())
        result = runner.forgetting_check()
        for key in ("base", "routed", "delta"):
            assert key in result["mmlu"]

    def test_saves_json_when_output_dir_set(self, mocker, tmp_path):
        self._patch_both_tasks(mocker, 0.7, 0.69)
        model = _make_model()
        runner = BenchmarkRunner(
            model, BenchmarkConfig(output_dir=str(tmp_path))
        )
        runner.forgetting_check()
        assert (tmp_path / "forgetting_check.json").exists()


# ── TestTaskRegistry ──────────────────────────────────────────────────────────


class TestTaskRegistry:
    def test_all_expected_tasks_registered(self):
        for name in ("mmlu", "hellaswag", "generation"):
            assert name in _TASK_REGISTRY

    def test_registry_values_are_task_subclasses(self):
        from concadptr.benchmarks.tasks import BenchmarkTask

        for cls in _TASK_REGISTRY.values():
            assert issubclass(cls, BenchmarkTask)

    def test_task_name_attributes_match_registry_keys(self):
        for key, cls in _TASK_REGISTRY.items():
            assert cls.name == key


# ── TestBenchmarkConfig ───────────────────────────────────────────────────────


class TestBenchmarkConfig:
    def test_defaults(self):
        config = BenchmarkConfig()
        assert "mmlu" in config.tasks
        assert "hellaswag" in config.tasks
        assert config.num_samples is None
        assert config.batch_size == 4

    def test_custom_values(self):
        config = BenchmarkConfig(num_samples=50, mmlu_subjects=["anatomy"])
        assert config.num_samples == 50
        assert config.mmlu_subjects == ["anatomy"]


# ── TestBenchmarkResult ───────────────────────────────────────────────────────


class TestBenchmarkResult:
    def test_fields_accessible(self):
        result = BenchmarkResult(
            task="mmlu",
            adapter_name="base",
            metrics={"accuracy": 0.72},
            num_samples=100,
            elapsed_seconds=5.3,
        )
        assert result.task == "mmlu"
        assert result.routing_stats is None

    def test_routing_stats_optional(self):
        result = BenchmarkResult(
            task="hellaswag",
            adapter_name="concadptr_routed",
            metrics={"accuracy": 0.80},
            num_samples=50,
            elapsed_seconds=3.0,
            routing_stats={"routing_entropy": 0.9},
        )
        assert result.routing_stats["routing_entropy"] == pytest.approx(0.9)
