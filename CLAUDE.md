# ConcAdptr — Concocting Adapters

Mixture-of-LoRA-Experts library. Takes independently trained LoRA adapters and concocts them into MoE-style expert systems with learned routing. Full pipeline: Train → Concoct → Serve.

## Quick Facts

- **Language**: Python 3.9+
- **Core deps**: PyTorch, HuggingFace Transformers, PEFT, safetensors
- **Package name**: `concadptr` (PyPI)
- **License**: Apache 2.0
- **Author**: Irfan Ali (@irfanalii)

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_core.py::TestXLoRARouter -v

# Type check
mypy concadptr/

# Lint
ruff check concadptr/

# Lint + fix
ruff check --fix concadptr/

# Build package
python -m build

# Publish to PyPI (after building)
twine upload dist/*
```

## Project Structure

```
concadptr/              # Main package
├── __init__.py        # Public API — all user-facing classes exported here
├── config.py          # ConcAdptrConfig, RouterConfig, TrainingConfig, ServingConfig
├── model.py           # ConcAdptrModel — core class (base model + adapters + router)
├── trainer.py         # ConcAdptrTrainer — router training loop
├── router/            # Routing strategies
│   ├── base.py        # BaseRouter ABC — all routers inherit from this
│   ├── soft_merging.py  # SoftMergingRouter (MoLoRA-style, dense, all experts)
│   ├── top_k.py       # TopKRouter (MixLoRA-style, sparse, top-k experts)
│   └── xlora.py       # XLoRARouter (X-LoRA-style, frozen adapters + learned scaling)
├── adapters/
│   └── __init__.py    # AdapterRegistry — register, validate, manage LoRA adapters
├── serving/
│   └── server.py      # FastAPI inference server (/v1/completions, /v1/adapters, /health)
└── utils/
    └── visualization.py  # Routing heatmaps, expert load plots, summary stats
tests/
└── test_core.py       # Unit tests for config, all 3 routers, registry, load balance
examples/
└── config.yaml        # Example YAML configuration
```

## Architecture & Key Concepts

The library has three main layers:

1. **Adapters** (`AdapterRegistry`): Register and validate multiple PEFT LoRA adapters. Checks rank, target modules, and base model compatibility.
2. **Router** (`BaseRouter` subclasses): Learns how to weight/select adapters for each input token. Three strategies: soft merging, top-k, X-LoRA.
3. **Model** (`ConcAdptrModel`): Ties base model + adapters + router together. Coordinates forward pass with weighted expert fusion.

The forward pass works like this:
- Each adapter produces its own logits from the input
- The router takes hidden states and produces per-expert weights
- Adapter logits are combined using routing weights
- Load-balancing loss discourages expert collapse

## Code Style

- Raw PyTorch — no HuggingFace Trainer or PyTorch Lightning
- Type hints on all public methods
- Docstrings in Google style on all public classes and methods
- `from __future__ import annotations` at the top of every module
- Imports: stdlib → third-party → local, separated by blank lines
- Line length: 100 chars (configured in pyproject.toml via ruff)
- No wildcard imports
- Dataclasses for config objects, not Pydantic (zero extra deps)

## Design Principles

- **Model-agnostic**: Works with any HuggingFace CausalLM. No model-specific code paths.
- **Adapters are frozen by default**: Router training should NOT modify adapter weights. Only the router parameters are trainable.
- **Base model is always frozen**: The pre-trained weights never change.
- **PEFT-compatible**: Adapters must be standard PEFT format (adapter_config.json + adapter_model.safetensors).
- **Privacy-first**: The entire design assumes customer data cannot leave their environment. Only adapter files (50-200MB) are transferred.
- **Consumer GPU friendly**: Default config uses 4-bit quantization (QLoRA/bitsandbytes). Must work on 16GB VRAM.

## Key TODOs

**Completed:**
- ~~Per-layer routing hook~~ — 2-pass forward with per-layer LoRA delta weighting via hooks
- ~~HuggingFace Hub upload/download~~ — `push_to_hub`, `from_hub`, `push_adapter_to_hub`, `load_adapter_from_hub`

**Serving:**
- vLLM integration for high-throughput multi-LoRA serving (§6.2 in research paper)
- Hook per-layer routing into the generation loop (currently only forward pass, not `generate()`)

**Adapter Merging (Static Fallback):**
- Linear weighted merging — weighted average of adapter deltas (§5.1)
- TIES-Merging — trim, elect sign, merge to reduce interference (§5.2)
- DARE — stochastic drop + rescale before merging (§5.3)
- Recommended: implement via `mergekit` integration or native PyTorch

**Evaluation & Benchmarking:**
- Benchmarking suite across model families (Qwen2.5, LLaMA 3.1, Mistral) (§9.3)
- Task-specific metrics (accuracy, F1, BLEU/ROUGE)
- General capability benchmarks to detect catastrophic forgetting (MMLU, HellaSwag)
- Per-adapter A/B comparison tooling

**Adapter Lifecycle:**
- Adapter version metadata (base model version, training config hash, eval metrics) (§7.3)
- Progressive merging pipeline — incremental adapter integration with quality gating (§7.1)

**Advanced (Future):**
- Federated LoRA training (FedAvg on adapter weights) (§11)
- Adapter distillation — compress multiple adapters into one lower-rank adapter (§11)
- Differential privacy training support (DP-SGD via Opacus) (§8.2)

## Testing

Tests use pytest. All router tests create small tensors (hidden_size=64) and verify:
- Output shapes are correct
- Routing weights sum to 1 (for soft/xlora) or have exactly k non-zero entries (for top-k)
- Load balance loss is low for uniform routing, high for collapsed routing
- Routing history recording works when enabled
- AdapterRegistry validates compatibility and catches mismatches

When adding a new router strategy, add tests following the pattern in `TestSoftMergingRouter`.

## Git Workflow

- Branch from `main` for features: `feature/per-layer-hooks`, `feature/vllm-integration`
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Run `pytest` and `ruff check` before pushing
- Tag releases as `v0.1.0`, `v0.2.0`, etc.
