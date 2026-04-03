# 🧪 ConcAdpt

**Concocting Adapters — Brew multiple LoRA adapters into Mixture-of-Experts systems.**

**Train → Concoct → Serve.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/concadpt/)

---

ConcAdpt takes independently trained LoRA adapters and concocts them into MoE-style expert systems with learned routing. Model-agnostic. Privacy-preserving. Built for production.

## The Problem

You fine-tune a base model with LoRA for your product. Then each customer needs their own specialization — but they can't share their data. You end up with multiple LoRA adapters trained in isolation. How do you combine them into something smarter than any individual adapter?

## The Solution

ConcAdpt takes your independently trained LoRA adapters and concocts them into experts within a Mixture-of-Experts system. A lightweight router learns which expert(s) to activate for each input — without needing access to any customer's original training data.

```
Base Model ─┬─ LoRA Adapter A (medical)    ──┐
            ├─ LoRA Adapter B (legal)      ──┼── Router ── Concocted Output
            └─ LoRA Adapter C (finance)    ──┘
```

## Key Features

- **Model-agnostic** — Works with any HuggingFace transformer (Qwen, LLaMA, Mistral, Gemma, etc.)
- **Privacy-preserving** — Customer data never leaves their environment; only adapters travel
- **3 routing strategies** — Soft merging (MoLoRA), Top-K sparse routing (MixLoRA), X-LoRA learned scaling
- **Full pipeline** — Train adapters → Concoct with router → Serve — one library
- **Production-ready** — FastAPI serving, adapter registry, compatibility validation
- **Consumer GPU friendly** — 4-bit quantization, runs on 16GB VRAM

## Installation

```bash
pip install concadpt
```

With optional dependencies:

```bash
pip install concadpt[training]   # + bitsandbytes, trl
pip install concadpt[serving]    # + fastapi, uvicorn
pip install concadpt[all]        # everything
```

## Quick Start

### 1. Define your configuration

```python
from concadpt import ConcAdptConfig

config = ConcAdptConfig(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    adapters={
        "medical": "./adapters/medical_invoices",
        "legal": "./adapters/legal_contracts",
        "finance": "./adapters/financial_reports",
    },
    routing_strategy="xlora",  # or "soft_merging", "top_k"
    quantization="4bit",
)
```

Or load from YAML:

```python
config = ConcAdptConfig.from_yaml("config.yaml")
```

### 2. Build the concocted model

```python
from concadpt import ConcAdptModel

model = ConcAdptModel.from_config(config)
print(model)
# ConcAdptModel(
#   base_model='Qwen/Qwen2.5-7B-Instruct',
#   num_adapters=3,
#   adapter_names=['medical', 'legal', 'finance'],
#   routing=xlora,
#   trainable_params=263,168
# )
```

### 3. Train the router

```python
from concadpt import ConcAdptTrainer
from datasets import load_dataset

# Use your general-purpose dataset (NOT customer data)
dataset = load_dataset("your_dataset")

trainer = ConcAdptTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=4,
)

results = trainer.train()
model.save_pretrained("./concocted_model")
```

### 4. Analyze routing patterns

```python
from concadpt.utils import print_routing_summary

model.router.enable_history(True)
# Run some inference...
stats = model.router.get_routing_stats()
print_routing_summary(stats, expert_names=["medical", "legal", "finance"])
```

Output:
```
ConcAdpt Routing Summary
========================================
Routing Entropy: 1.0234 / 1.0986 (max)
Uniformity: 93.2%

Expert Load:
  medical              0.3841 ███████████████████
  legal                0.3012 ███████████████
  finance              0.3147 ███████████████

Expert Utilization (top-2):
  medical              0.8234 █████████████████████████████████████████
  legal                0.6891 ██████████████████████████████████
  finance              0.7102 ███████████████████████████████████
```

### 5. Serve (optional)

```python
from concadpt.serving import serve

serve("./concocted_model", host="0.0.0.0", port=8000)
```

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Analyze this medical invoice...", "max_tokens": 256}'
```

## Routing Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| `soft_merging` | Weighted average of ALL experts per token | Few experts (2-8), overlapping domains |
| `top_k` | Activate only top-k experts per token | Many experts (8+), distinct domains |
| `xlora` | Learned scaling with frozen adapters, layer-wise | Independent adapters, privacy-critical |

## Architecture

```
┌──────────────────────────────────────────────┐
│                 ConcAdptModel                │
│                                              │
│  ┌──────────┐  ┌───────────────────────────┐ │
│  │   Base    │  │     Adapter Registry      │ │
│  │  Model    │  │  ┌─────┐ ┌─────┐ ┌─────┐ │ │
│  │ (frozen)  │  │  │LoRA │ │LoRA │ │LoRA │ │ │
│  │           │  │  │  A  │ │  B  │ │  C  │ │ │
│  │           │  │  │froze│ │froze│ │froze│ │ │
│  └──────────┘  │  └──┬──┘ └──┬──┘ └──┬──┘ │ │
│                │     │       │       │     │ │
│                └─────┼───────┼───────┼─────┘ │
│                      │       │       │       │
│                ┌─────▼───────▼───────▼─────┐ │
│                │         Router            │ │
│                │       (trainable)         │ │
│                │  ┌──────────────────────┐ │ │
│                │  │   Gating Network     │ │ │
│                │  └──────────┬───────────┘ │ │
│                └─────────────┼─────────────┘ │
│                              │               │
│                    ┌─────────▼─────────┐     │
│                    │ Concocted Output  │     │
│                    └───────────────────┘     │
└──────────────────────────────────────────────┘
```

## The Multi-Customer Use Case

ConcAdpt was designed for a specific real-world pattern:

1. **You** fine-tune a base model on your general training data → your product's foundation model
2. **Each customer** fine-tunes on their private data (on-premise) → produces a LoRA adapter
3. **The adapter** (50-200MB, no raw data) is transferred back to you
4. **ConcAdpt** concocts all customer adapters into a MoE system with learned routing

Customer data never leaves their environment. The router learns which expert(s) to activate without seeing the original training data. This is **federated expertise** — cross-customer knowledge transfer without data sharing.

## Project Structure

```
concadpt/
├── concadpt/
│   ├── __init__.py          # Public API
│   ├── config.py            # Configuration classes
│   ├── model.py             # ConcAdptModel (core)
│   ├── trainer.py           # ConcAdptTrainer (router training)
│   ├── router/
│   │   ├── base.py          # BaseRouter ABC
│   │   ├── soft_merging.py  # Dense/soft routing (MoLoRA)
│   │   ├── top_k.py         # Sparse top-k routing (MixLoRA)
│   │   └── xlora.py         # Learned scaling (X-LoRA)
│   ├── adapters/
│   │   └── __init__.py      # AdapterRegistry
│   ├── serving/
│   │   └── server.py        # FastAPI inference server
│   └── utils/
│       └── visualization.py # Routing analysis tools
├── tests/
│   └── test_core.py         # Unit tests
├── examples/
│   └── config.yaml          # Example configuration
├── pyproject.toml
├── LICENSE
└── README.md
```

## Development

```bash
git clone https://github.com/irfanalii/concadpt.git
cd concadpt
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Core library architecture
- [x] 3 routing strategies (soft, top-k, X-LoRA)
- [x] Adapter registry with compatibility validation
- [x] Router training pipeline
- [x] FastAPI serving
- [x] Routing visualization and analysis
- [ ] Full generation loop with per-layer routing hooks
- [ ] Integration with vLLM for high-throughput serving
- [ ] Adapter merging (linear, TIES, DARE) as fallback
- [ ] Hugging Face Hub adapter upload/download
- [ ] Benchmarking suite across model families
- [ ] Distributed router training

## References

- Hu et al. (2021) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Zadouri et al. (2023) — [Pushing Mixture of Experts to the Limit (MoLoRA)](https://arxiv.org/abs/2309.05444)
- Wu et al. (2024) — [Mixture of LoRA Experts (MoLE)](https://arxiv.org/abs/2404.13628)
- Li et al. (2024) — [MixLoRA](https://arxiv.org/abs/2404.15159)
- Buehler & Buehler (2024) — [X-LoRA: Mixture of Low-Rank Adapter Experts](https://github.com/EricLBuehler/xlora)
- Zhuang et al. (2025) — [LD-MoLE: Learnable Dynamic Routing for MoLE](https://arxiv.org/abs/2509.25684)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Author

**Irfan Ali** — [GitHub](https://github.com/irfanalii) · [HuggingFace](https://huggingface.co/irfanalii) · [LinkedIn](https://linkedin.com/in/irfanalii)

---

*ConcAdpt — because the best models are concocted, not just trained.*
