"""Tests for ProgressiveMerger and QualityGateError."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from concadptr.merging.progressive import (
    MergeResult,
    ProgressiveMerger,
    ProgressiveMergerConfig,
    QualityGateError,
)

_RUNNER_PATH = "concadptr.merging.progressive.BenchmarkRunner"
_MERGE_PATH = "concadptr.merging.merge_adapters"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _good_forgetting(tasks=("mmlu", "hellaswag")):
    """Forgetting check result where all deltas pass the default gate."""
    return {t: {"base": 0.70, "routed": 0.69, "delta": -0.01} for t in tasks}


def _bad_forgetting(bad_task="mmlu", tasks=("mmlu", "hellaswag")):
    """Forgetting check result where one task fails the default gate (-0.02)."""
    result = {t: {"base": 0.70, "routed": 0.69, "delta": -0.01} for t in tasks}
    result[bad_task] = {"base": 0.70, "routed": 0.65, "delta": -0.05}
    return result


def _make_merger(mock_model, config=None):
    return ProgressiveMerger(mock_model, config or ProgressiveMergerConfig())


# ── QualityGateError ──────────────────────────────────────────────────────────


class TestQualityGateError:
    def test_stores_deltas(self):
        deltas = {"mmlu": -0.05, "hellaswag": -0.01}
        err = QualityGateError("failed", deltas=deltas)
        assert err.deltas is deltas

    def test_is_exception(self):
        with pytest.raises(QualityGateError):
            raise QualityGateError("msg", deltas={})


# ── ProgressiveMergerConfig ───────────────────────────────────────────────────


class TestProgressiveMergerConfig:
    def test_defaults(self):
        cfg = ProgressiveMergerConfig()
        assert cfg.quality_gate_threshold == -0.02
        assert cfg.merge_method == "linear"
        assert cfg.merge_weights is None
        assert cfg.benchmark_num_samples == 50
        assert cfg.benchmark_tasks == ["mmlu", "hellaswag"]

    def test_custom_threshold(self):
        cfg = ProgressiveMergerConfig(quality_gate_threshold=-0.05)
        assert cfg.quality_gate_threshold == -0.05


# ── ProgressiveMerger ─────────────────────────────────────────────────────────


class TestProgressiveMerger:
    @pytest.fixture
    def mock_model(self):
        """Minimal mock ConcAdptrModel with a real AdapterRegistry."""
        from concadptr.adapters import AdapterInfo, AdapterRegistry

        model = MagicMock()
        registry = AdapterRegistry()
        registry._adapters = {
            "a": AdapterInfo(name="a", path="/fake/a", rank=16),
            "b": AdapterInfo(name="b", path="/fake/b", rank=16),
        }
        model.registry = registry
        return model

    # -- gate passes → merges -------------------------------------------------

    def test_add_adapter_passes_gate(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH) as mock_merge:
            MockRunner.return_value.forgetting_check.return_value = _good_forgetting()
            mock_merge.return_value = tmp_path / "merged"

            merger = _make_merger(mock_model)
            result = merger.add_adapter("new", path, tmp_path / "merged")

        assert result.passed_gate is True
        assert result.merged is True
        assert result.adapter_name == "new"
        mock_merge.assert_called_once()

    # -- gate fails → raises, unregisters ------------------------------------

    def test_add_adapter_fails_gate(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH) as mock_merge:
            MockRunner.return_value.forgetting_check.return_value = _bad_forgetting()

            merger = _make_merger(mock_model)
            with pytest.raises(QualityGateError) as exc_info:
                merger.add_adapter("new", path, tmp_path / "merged")

        mock_merge.assert_not_called()
        assert "new" not in mock_model.registry
        assert "mmlu" in exc_info.value.deltas

    # -- dry_run → gate passes, no merge -------------------------------------

    def test_add_adapter_dry_run(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH) as mock_merge:
            MockRunner.return_value.forgetting_check.return_value = _good_forgetting()

            merger = _make_merger(mock_model)
            result = merger.add_adapter("new", path, tmp_path / "merged", dry_run=True)

        assert result.passed_gate is True
        assert result.merged is False
        mock_merge.assert_not_called()

    # -- dry_run + gate failure still raises ---------------------------------

    def test_dry_run_gate_failure_still_raises(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH):
            MockRunner.return_value.forgetting_check.return_value = _bad_forgetting()

            merger = _make_merger(mock_model)
            with pytest.raises(QualityGateError):
                merger.add_adapter("new", path, tmp_path / "merged", dry_run=True)

    # -- merge_method forwarded to merge_adapters ----------------------------

    def test_merge_method_passed_through(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        cfg = ProgressiveMergerConfig(merge_method="ties")
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH) as mock_merge:
            MockRunner.return_value.forgetting_check.return_value = _good_forgetting()
            mock_merge.return_value = tmp_path / "merged"

            _make_merger(mock_model, cfg).add_adapter("new", path, tmp_path / "merged")

        assert mock_merge.call_args.kwargs["method"] == "ties"

    # -- equal weights when merge_weights is None ----------------------------

    def test_equal_weights_when_none(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH) as mock_merge:
            MockRunner.return_value.forgetting_check.return_value = _good_forgetting()
            mock_merge.return_value = tmp_path / "merged"

            _make_merger(mock_model).add_adapter("new", path, tmp_path / "merged")

        # registry has a, b + new = 3 adapters
        weights = mock_merge.call_args.kwargs["weights"]
        assert len(weights) == 3
        assert all(abs(w - 1 / 3) < 1e-9 for w in weights)

    # -- custom weights forwarded --------------------------------------------

    def test_custom_weights_forwarded(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        cfg = ProgressiveMergerConfig(merge_weights=[0.5, 0.3, 0.2])
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH) as mock_merge:
            MockRunner.return_value.forgetting_check.return_value = _good_forgetting()
            mock_merge.return_value = tmp_path / "merged"

            _make_merger(mock_model, cfg).add_adapter("new", path, tmp_path / "merged")

        assert mock_merge.call_args.kwargs["weights"] == [0.5, 0.3, 0.2]

    # -- adapter is registered before forgetting_check is called -------------

    def test_registers_adapter_before_check(self, mock_model, adapter_dir, tmp_path):
        path = adapter_dir("new")
        seen_names: list[list[str]] = []

        def capture_forgetting():
            seen_names.append(list(mock_model.registry.names))
            return _good_forgetting()

        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH):
            MockRunner.return_value.forgetting_check.side_effect = capture_forgetting

            _make_merger(mock_model).add_adapter("new", path, tmp_path / "merged")

        assert "new" in seen_names[0]

    # -- FileNotFoundError propagates from registry -------------------------

    def test_invalid_path_propagates(self, mock_model, tmp_path):
        merger = _make_merger(mock_model)
        with pytest.raises(FileNotFoundError):
            merger.add_adapter("ghost", tmp_path / "nonexistent", tmp_path / "out")

    # -- gate fails on ANY task (not just average) ---------------------------

    def test_multiple_tasks_any_fail(self, mock_model, adapter_dir, tmp_path):
        """Gate must reject when only one task exceeds the threshold."""
        path = adapter_dir("new")
        forgetting = {
            "mmlu": {"base": 0.70, "routed": 0.69, "delta": -0.01},        # passes
            "hellaswag": {"base": 0.80, "routed": 0.74, "delta": -0.06},   # fails
        }
        with patch(_RUNNER_PATH) as MockRunner, patch(_MERGE_PATH):
            MockRunner.return_value.forgetting_check.return_value = forgetting

            merger = _make_merger(mock_model)
            with pytest.raises(QualityGateError) as exc_info:
                merger.add_adapter("new", path, tmp_path / "merged")

        assert "hellaswag" in exc_info.value.deltas
        assert exc_info.value.deltas["hellaswag"] == pytest.approx(-0.06)
