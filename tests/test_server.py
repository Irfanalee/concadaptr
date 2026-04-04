"""Tests for the FastAPI inference server.

Strategy: patch ConcAdptrModel.load_pretrained to return a pre-built mock so
no HuggingFace model is loaded.  Wrap TestClient inside the patch context so
the startup event runs with the mock in scope.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (required by TestClient)")

from fastapi.testclient import TestClient

from concadptr.serving.server import create_app


# ── Mock model factory ──


def _make_mock_model(adapter_names=("medical", "legal")):
    """Return a MagicMock that quacks like a loaded ConcAdptrModel."""
    mock = MagicMock()
    mock.registry.names = list(adapter_names)
    mock.registry.num_adapters = len(adapter_names)
    mock.config.router.strategy.value = "xlora"
    mock.router = MagicMock()
    mock.router.parameters.return_value = iter([])
    mock.tokenizer = MagicMock()

    import torch
    fake_inputs = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    mock.tokenizer.return_value = fake_inputs

    return mock


# ── Fixture ──


@pytest.fixture
def client():
    """TestClient with a mocked model loaded on startup."""
    mock_model = _make_mock_model()

    with patch(
        "concadptr.serving.server.ConcAdptrModel",  # patched inside create_app's startup
        autospec=False,
    ) as _MockClass:
        # Patch the class used inside the startup handler
        pass  # we'll patch differently below

    # The startup handler imports ConcAdptrModel at call time, so patch there
    target = "concadptr.model.ConcAdptrModel.load_pretrained"
    with patch(target, return_value=mock_model):
        app = create_app(model_path="/fake/model")
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c, mock_model


# ── /health ──


class TestHealthEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, client):
        c, _ = client
        data = resp = c.get("/health").json()
        assert data["status"] == "ok"

    def test_model_loaded_true(self, client):
        c, _ = client
        data = c.get("/health").json()
        assert data["model_loaded"] is True


# ── /v1/adapters ──


class TestAdaptersEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.get("/v1/adapters")
        assert resp.status_code == 200

    def test_adapter_list_matches_mock(self, client):
        c, mock = client
        data = c.get("/v1/adapters").json()
        assert set(data["adapters"]) == set(mock.registry.names)

    def test_num_adapters_correct(self, client):
        c, mock = client
        data = c.get("/v1/adapters").json()
        assert data["num_adapters"] == mock.registry.num_adapters

    def test_routing_strategy_present(self, client):
        c, _ = client
        data = c.get("/v1/adapters").json()
        assert "routing_strategy" in data


# ── /v1/completions ──


class TestCompletionsEndpoint:
    def test_returns_200(self, client):
        c, _ = client
        resp = c.post("/v1/completions", json={"prompt": "Hello"})
        assert resp.status_code == 200

    def test_response_has_text_field(self, client):
        c, _ = client
        data = c.post("/v1/completions", json={"prompt": "Hello"}).json()
        assert "text" in data

    def test_response_has_routing_weights(self, client):
        c, _ = client
        data = c.post("/v1/completions", json={"prompt": "Hello"}).json()
        assert "routing_weights" in data

    def test_response_has_tokens_generated(self, client):
        c, _ = client
        data = c.post("/v1/completions", json={"prompt": "Hello"}).json()
        assert "tokens_generated" in data

    def test_routing_weights_keys_match_adapters(self, client):
        c, mock = client
        data = c.post("/v1/completions", json={"prompt": "Hello"}).json()
        assert set(data["routing_weights"].keys()) == set(mock.registry.names)

    def test_custom_max_tokens_accepted(self, client):
        c, _ = client
        resp = c.post("/v1/completions", json={"prompt": "Hi", "max_tokens": 512})
        assert resp.status_code == 200

    def test_missing_prompt_returns_422(self, client):
        c, _ = client
        resp = c.post("/v1/completions", json={})
        assert resp.status_code == 422
