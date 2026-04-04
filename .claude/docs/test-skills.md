# ConcAdptr Testing Guide

ConcAdptr is a pure Python/PyTorch ML library — no frontend, no database, no
external service dependencies. This shapes which test types matter and which don't.

---

## Test File Map

```
tests/
├── conftest.py         → shared fixtures (adapter_dir, tiny_router,
│                          mock_concadptr_model, make_mock_base_model)
├── test_config.py      → TestConcAdptrConfig
├── test_routers.py     → TestSoftMergingRouter, TestTopKRouter, TestXLoRARouter,
│                          TestLoadBalanceLoss, TestRoutingStats
├── test_adapters.py    → TestAdapterRegistry
├── test_model.py       → TestConcAdptrModelForward, TestGradientFreeze,
│                          TestSavePretrained
├── test_trainer.py     → TestConcAdptrTrainer
└── test_server.py      → TestHealthEndpoint, TestAdaptersEndpoint,
                           TestCompletionsEndpoint
```

---

## Layer Map

| Layer | Technology | Testable concern |
|---|---|---|
| Router math | PyTorch `nn.Module` | Tensor shapes, weight constraints, gradient flow |
| Config system | Python dataclasses | Validation logic, YAML serialisation |
| Adapter registry | Pure Python + filesystem | File parsing, compatibility rules |
| Model composition | PyTorch + PEFT | Forward pass correctness, adapter dispatch |
| Training loop | Raw PyTorch | Step counting, optimiser behaviour |
| Inference server | FastAPI | HTTP schema, startup, routing |

---

## `conftest.py` Fixtures

All fixtures live in `tests/conftest.py` and are available to every test file
without importing.

| Fixture | Returns | Use when |
|---|---|---|
| `adapter_dir` | `Callable(name, rank, base_model) → Path` | Any test that needs a real PEFT adapter directory on disk |
| `tiny_router` | `SoftMergingRouter(hidden=64, experts=2, layers=2)` | Quick router math checks |
| `mock_concadptr_model` | `ConcAdptrModel` with injected mocks | Forward pass / gradient tests without HF downloads |
| `make_mock_base_model()` | `MagicMock` shaped like `AutoModelForCausalLM` | Building custom model mocks in `test_model.py` |

`adapter_dir` is a **factory fixture** — call it like a function inside the test:

```python
def test_register(adapter_dir):
    d = adapter_dir("medical", rank=16)   # → tmp_path/medical/
    registry.register("medical", d)
```

---

## Test Types: What Applies and Why

### ✅ Unit Testing — Core

**What**: Test a single class or function in isolation, no I/O.

**Why it fits**: Router math (softmax, top-k, load-balance loss) is pure tensor
operations. Config validation is pure Python. Both are fast and hermetic.

**Examples in this repo**:
- `tests/test_routers.py::TestSoftMergingRouter::test_weights_sum_to_one`
- `tests/test_config.py::TestConcAdptrConfig::test_validation_invalid_quantization`
- `tests/test_routers.py::TestLoadBalanceLoss::test_collapsed_routing_higher_than_uniform`

**New router checklist** — for every new routing strategy, add a class to
`tests/test_routers.py` with at least these five tests:

```python
class TestMyNewRouter:
    def test_output_shape(self): ...             # shape == (batch, seq, num_experts)
    def test_weights_sum_to_one(self): ...       # or sparsity for top-k variants
    def test_requires_layer_idx(self): ...       # if use_layerwise=True
    def test_load_balance_loss_uniform(self): ...# loss ≈ 1.0 for uniform input
    def test_history_recording(self): ...        # enable_history(True) → check stats
```

---

### ✅ Integration Testing — High Value

**What**: Test the boundary between ConcAdptr and PEFT — real `set_adapter`,
`load_adapter`, and `PeftModel` calls that mocks can't replicate.

**Why it fits**: `test_model.py` mocks the base model, so it never touches actual
PEFT. The real PEFT code enforces adapter name registration, weight shape
compatibility, and config presence — a mock accepts anything.

**Status**: **Not yet written.** Add to `tests/test_model.py` marked `@pytest.mark.slow`.

```python
# tests/test_model.py (to be added)
@pytest.mark.slow
def test_peft_adapter_loads_and_routes(tmp_path):
    from transformers import AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig
    # Use a tiny real model (distilgpt2) + real PEFT adapter
    # Then ConcAdptrModel.from_config(...) — exercises full PEFT boundary
```

---

### ✅ API Endpoint Testing — Medium Value

**What**: Spin up the FastAPI app in-process with `TestClient`, send HTTP requests.

**Why it fits**: Pydantic request/response schemas and the startup lifecycle are
real FastAPI behaviour not exercised by unit tests.

**Where**: `tests/test_server.py` — `TestHealthEndpoint`, `TestAdaptersEndpoint`,
`TestCompletionsEndpoint`.

**Mock strategy**: patch `concadptr.model.ConcAdptrModel.load_pretrained` so no
model is downloaded. The fixture in `test_server.py` wraps `TestClient` inside
the patch context so the `startup` event fires with the mock in scope:

```python
with patch("concadptr.model.ConcAdptrModel.load_pretrained", return_value=mock_model):
    app = create_app(model_path="/fake/model")
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"
```

---

### ✅ Save/Load Roundtrip — High Value

**What**: `save_pretrained` → `load_pretrained` → verify router weights are
identical.

**Why it fits**: A bug here (wrong key name, missing file, wrong `map_location`)
silently discards all trained router weights at serving time.

**Where**: `tests/test_model.py::TestSavePretrained::test_router_weights_roundtrip`

**How**: fill all router params with 0.42, save, reload via monkeypatched
`from_config`, compare tensors with `torch.allclose`.

---

### ✅ Gradient Flow Testing — High Value

**What**: Backward pass → assert router params have gradients, adapter params don't.

**Why it fits**: `freeze_adapters=True` is the core design invariant. A subtle
`requires_grad=True` leak in `_load_adapters` would silently corrupt adapter
weights during router training.

**Where**: `tests/test_model.py::TestGradientFreeze`

```python
# Verify router gets gradients
for param in model.router.parameters():
    assert param.grad is not None and param.grad.abs().sum() > 0
```

---

### ✅ Property-Based Testing — Nice to Have

**What**: Use `hypothesis` to generate random `(batch, seq, hidden)` combos and
assert invariants hold.

**Why it fits**: Edge cases (seq_len=1, batch=32, hidden=4096) expose silent shape
bugs that hand-crafted tests miss.

**Status**: Not yet written. Would live in `tests/test_routers.py`.

```python
from hypothesis import given, settings
from hypothesis import strategies as st

@given(batch=st.integers(1, 4), seq=st.integers(1, 32),
       hidden=st.sampled_from([32, 64, 128]))
@settings(max_examples=30)
def test_soft_router_always_sums_to_one(batch, seq, hidden):
    router = SoftMergingRouter(hidden_size=hidden, num_experts=3, num_layers=1)
    weights = router(torch.randn(batch, seq, hidden), layer_idx=0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch, seq), atol=1e-5)
```

---

## Test Types That Don't Apply

| Test Type | Why |
|---|---|
| **UI / browser testing** | No frontend. |
| **Database testing** | No database or ORM. |
| **Contract testing** | No service-to-service consumer boundary. |
| **Load / stress testing** | `/v1/completions` generation is a placeholder — test once implemented. |
| **Security / pen testing** | No auth surface, no SQL. `bandit` as a CI lint step is sufficient. |
| **Visual snapshot testing** | `matplotlib` PNG output is brittle across OS/font versions. Test tensor values instead. |
| **Mutation testing** | Premature until suite is substantially complete. Revisit at >80% coverage. |

---

## Priority Tiers

### Tier 1 — Blocking (must pass before any merge)

| Test | Class | File |
|---|---|---|
| Forward pass output keys, shapes, loss with/without labels | `TestConcAdptrModelForward` | `test_model.py` |
| `set_adapter` called once per adapter per forward | `TestConcAdptrModelForward` | `test_model.py` |
| `save_pretrained` / `load_pretrained` roundtrip | `TestSavePretrained` | `test_model.py` |
| Router params have gradients, adapter params don't | `TestGradientFreeze` | `test_model.py` |

### Tier 2 — High Value

| Test | Class | File |
|---|---|---|
| `global_step == floor(num_batches / accum_steps)` | `TestConcAdptrTrainer` | `test_trainer.py` |
| `/health` returns `{status, model_loaded}` | `TestHealthEndpoint` | `test_server.py` |
| `/v1/adapters` returns correct adapter list | `TestAdaptersEndpoint` | `test_server.py` |
| `/v1/completions` returns expected schema | `TestCompletionsEndpoint` | `test_server.py` |

### Tier 3 — Complete Coverage

| Test | Class | File |
|---|---|---|
| TOP_K k > num_adapters warning | `TestConcAdptrConfig` | `test_config.py` |
| Invalid quantization error | `TestConcAdptrConfig` | `test_config.py` |
| `FileNotFoundError` for missing adapter path | `TestAdapterRegistry` | `test_adapters.py` |
| XLoRA `get_layer_scalings` shape (layerwise + non-layerwise) | `TestXLoRARouter` | `test_routers.py` |

---

## Running Tests

```bash
# All tests (no GPU, no internet required)
pytest tests/ -v

# Skip slow integration tests
pytest tests/ -m "not slow"

# Skip GPU tests
pytest tests/ -m "not gpu"

# Single file
pytest tests/test_routers.py -v

# Single class
pytest tests/test_model.py::TestConcAdptrModelForward -v

# With coverage
pytest tests/ --cov=concadptr --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=concadptr --cov-report=html && open htmlcov/index.html

# Lint + tests together
ruff check concadptr/ tests/ && pytest tests/
```
