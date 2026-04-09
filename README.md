# ConcAdptr

**Concocting Adapters — Brew multiple LoRA adapters into Mixture-of-Experts systems.**

**Train → Concoct → Serve.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-orange.svg)](https://pypi.org/project/concadptr/)

---

ConcAdptr takes independently trained LoRA adapters and concocts them into MoE-style expert systems with learned routing. Model-agnostic. Privacy-preserving. Built for production.

## The Problem

You fine-tune a base model with LoRA for your product. Then each customer or user-group needs their own specialization — but they can't share their data. You end up with multiple LoRA adapters trained in isolation. How do you combine them into something smarter than any individual adapter?

## The Solution

ConcAdptr takes your independently trained LoRA adapters and concocts them into experts within a Mixture-of-Experts system. A lightweight router learns which expert(s) to activate for each input — without needing access to any customer's original training data.

```
Base Model ─┬─ LoRA Adapter A (medical)    ──┐
            ├─ LoRA Adapter B (legal)      ──┼── Router ── Concocted Output
            └─ LoRA Adapter C (finance)    ──┘
```

## Key Features

- **Model-agnostic** — Works with any HuggingFace transformer (Qwen, LLaMA, Mistral, Gemma, etc.)
- **Privacy-preserving** — Customer data never leaves their environment; only adapters travel
- **3 routing strategies** — Soft merging (MoLoRA), Top-K sparse routing (MixLoRA), X-LoRA learned scaling
- **Routing in generation** — Per-layer routing hooks active during `model.generate()`, not just training
- **Static merging fallback** — Linear, TIES, and DARE merging when routing overhead is undesirable
- **Benchmarking suite** — MMLU, HellaSwag, BLEU/ROUGE; A/B adapter comparison and forgetting detection
- **Adapter version metadata** — track version, training config hash, and eval metrics per adapter
- **HuggingFace Hub integration** — Push/pull adapters and full models to/from the Hub
- **Full pipeline** — Train adapters → Concoct with router → Serve — one library
- **Production-ready** — FastAPI serving, adapter registry, compatibility validation
- **Consumer GPU friendly** — 4-bit quantization, runs on 16GB VRAM

## Installation

```bash
pip install concadptr
```

With optional dependencies:

```bash
pip install concadptr[training]    # + bitsandbytes, trl
pip install concadptr[serving]     # + fastapi, uvicorn
pip install concadptr[hub]         # + huggingface_hub
pip install concadptr[benchmarks]  # + evaluate (for BLEU/ROUGE scoring)
pip install concadptr[all]         # everything
```

## Quick Start

### 1. Define your configuration

```python
from concadptr import ConcAdptrConfig

config = ConcAdptrConfig(
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
config = ConcAdptrConfig.from_yaml("config.yaml")
```

### 2. Build the concocted model

```python
from concadptr import ConcAdptrModel

model = ConcAdptrModel.from_config(config)
```

### 3. Train the router

```python
from concadptr import ConcAdptrTrainer
from datasets import load_dataset, concatenate_datasets

# Mix of domain samples — not customer data
router_dataset = concatenate_datasets([
    load_dataset("medical_qa", split="train[:500]"),
    load_dataset("legal_docs", split="train[:500]"),
    load_dataset("finance_qa", split="train[:500]"),
])

trainer = ConcAdptrTrainer(
    model=model,
    train_dataset=router_dataset,
    learning_rate=1e-4,
    num_epochs=3,
    batch_size=4,
)

trainer.train()
model.save_pretrained("./concocted_model")
```

### 4. Generate text

```python
# Routing hooks are active during generation — no extra setup needed
output_ids = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
```

The router runs once on the full prompt (Pass 1) to compute per-layer routing weights. Those weights are cached and applied as hooks at every generation step — compatible with KV-cache.

### 5. Analyze routing patterns

```python
from concadptr.utils import print_routing_summary

model.router.enable_history(True)
# Run some inference...
stats = model.router.get_routing_stats()
print_routing_summary(stats, expert_names=["medical", "legal", "finance"])
```

### 6. Serve

```python
from concadptr.serving import serve

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

## Static Merging (No Router)

When routing overhead is undesirable, merge adapters statically into a single PEFT adapter:

```python
from concadptr import merge_adapters

# Linear weighted average
output = merge_adapters(
    adapters={"medical": "./adapters/medical", "legal": "./adapters/legal"},
    output_path="./merged",
    method="linear",       # "linear", "ties", "dare", "dare_ties"
    weights=[0.6, 0.4],
)

# TIES — reduces interference between adapters
output = merge_adapters(adapters=..., output_path="./merged", method="ties", trim_fraction=0.2)

# DARE — stochastic drop + rescale before merging
output = merge_adapters(adapters=..., output_path="./merged", method="dare", density=0.7)
```

Or via the registry:

```python
registry.merge(["medical", "legal"], output_path="./merged", method="ties")
```

The output is a standard PEFT adapter directory — usable with `PeftModel.from_pretrained()`.

## Benchmarking

Evaluate adapter quality, detect catastrophic forgetting, and compare model variants side by side.

```python
from concadptr import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(
    tasks=["mmlu", "hellaswag"],
    num_samples=200,           # None = full dataset
    output_dir="./bench_out",  # optional JSON output
)
runner = BenchmarkRunner(model, config)
```

**Evaluate the routed model:**

```python
results = runner.run()
# [BenchmarkResult(task="mmlu", metrics={"accuracy": 0.71}, ...),
#  BenchmarkResult(task="hellaswag", metrics={"accuracy": 0.78}, ...)]
```

**Compare base model, each adapter, and full routing:**

```python
comparison = runner.compare()
# {
#   "base":              [BenchmarkResult(task="mmlu", ...), ...],
#   "medical":           [BenchmarkResult(task="mmlu", ...), ...],
#   "legal":             [...],
#   "concadptr_routed":  [...],
# }
```

**Detect catastrophic forgetting:**

```python
deltas = runner.forgetting_check()
# {
#   "mmlu":      {"base": 0.68, "routed": 0.67, "delta": -0.01},
#   "hellaswag": {"base": 0.79, "routed": 0.78, "delta": -0.01},
# }
```

**Custom generation task with BLEU/ROUGE** (requires `pip install concadptr[benchmarks]`):

```python
config = BenchmarkConfig(
    tasks=["generation"],
    generation_dataset="my_org/qa_dataset",
    generation_input_field="question",
    generation_reference_field="answer",
    generation_metrics=["bleu", "rouge"],
    num_samples=100,
)
results = runner.run()
# BenchmarkResult(metrics={"bleu": 34.1, "rouge1": 0.52, "rouge2": 0.28, "rougeL": 0.49})
```

## Adapter Version Metadata

Track provenance and quality for each adapter — version, training config hash, and eval metrics — stored as a sidecar `concadptr_version.json` alongside the adapter weights.

```python
from concadptr.adapters import AdapterInfo

# Hash a training config for reproducibility tracking
config_hash = AdapterInfo.compute_config_hash({
    "learning_rate": 1e-4,
    "num_epochs": 3,
    "batch_size": 4,
    "lora_rank": 16,
})

# Store eval results on a registered adapter
registry.set_eval_metrics("medical", {"mmlu": 0.72, "hellaswag": 0.75}, save=True)

# Read back
info = registry.get("medical")
print(info.version)              # "1.2.0"
print(info.eval_metrics)         # {"mmlu": 0.72, "hellaswag": 0.75}
print(info.training_config_hash) # "3a8f1c..."

# Persist manually after setting fields
info.version = "1.3.0"
info.training_config_hash = config_hash
info.save_version_metadata()
```

Version metadata is loaded automatically when you call `registry.register()` — no extra steps needed if `concadptr_version.json` is present in the adapter directory.

## HuggingFace Hub

```python
# Push a full concocted model
model.push_to_hub("username/my-concocted-model", token="hf_...")

# Load it back
model = ConcAdptrModel.from_hub("username/my-concocted-model")

# Push/pull individual adapters
registry.push_adapter_to_hub("medical", repo_id="username/medical-adapter")
registry.load_adapter_from_hub("username/medical-adapter", name="medical")
```

## Architecture

```
┌──────────────────────────────────────────────┐
│                 ConcAdptrModel               │
│                                              │
│  ┌──────────┐  ┌───────────────────────────┐ │
│  │   Base   │  │     Adapter Registry      │ │
│  │  Model   │  │  ┌─────┐ ┌─────┐ ┌─────┐ │ │
│  │ (frozen) │  │  │LoRA │ │LoRA │ │LoRA │ │ │
│  │          │  │  │  A  │ │  B  │ │  C  │ │ │
│  │          │  │  │froze│ │froze│ │froze│ │ │
│  └──────────┘  │  └──┬──┘ └──┬──┘ └──┬──┘ │ │
│                └─────┼───────┼───────┼─────┘ │
│                      │       │       │       │
│                ┌─────▼───────▼───────▼─────┐ │
│                │         Router            │ │
│                │       (trainable)         │ │
│                └─────────────┬─────────────┘ │
│                              │               │
│                    ┌─────────▼─────────┐     │
│                    │ Concocted Output  │     │
│                    └───────────────────┘     │
└──────────────────────────────────────────────┘
```

## The Multi-Customer Use Case

ConcAdptr was designed for a specific real-world pattern:

1. **You** fine-tune a base model on your general training data → your product's foundation model
2. **Each customer** fine-tunes on their private data (on-premise) → produces a LoRA adapter
3. **The adapter** (50-200MB, no raw data) is transferred back to you
4. **ConcAdptr** concocts all customer adapters into a MoE system with learned routing

Customer data never leaves their environment. The router learns which expert(s) to activate without seeing the original training data. This is **federated expertise** — cross-customer knowledge transfer without data sharing.

## Project Structure

```
concadptr/
├── concadptr/
│   ├── __init__.py          # Public API
│   ├── config.py            # Configuration classes (ConcAdptrConfig, MergeConfig, ...)
│   ├── model.py             # ConcAdptrModel (core)
│   ├── trainer.py           # ConcAdptrTrainer (router training)
│   ├── router/
│   │   ├── base.py          # BaseRouter ABC
│   │   ├── soft_merging.py  # Dense/soft routing (MoLoRA)
│   │   ├── top_k.py         # Sparse top-k routing (MixLoRA)
│   │   └── xlora.py         # Learned scaling (X-LoRA)
│   ├── adapters/
│   │   └── __init__.py      # AdapterRegistry, AdapterInfo (version metadata)
│   ├── benchmarks/
│   │   ├── __init__.py      # BenchmarkRunner, BenchmarkConfig, BenchmarkResult
│   │   ├── config.py        # BenchmarkConfig, BenchmarkResult dataclasses
│   │   ├── metrics.py       # accuracy, f1_score, bleu, rouge
│   │   ├── tasks.py         # MMLUTask, HellaSwagTask, GenerationTask
│   │   └── runner.py        # BenchmarkRunner
│   ├── merging/
│   │   ├── __init__.py      # merge_adapters() functional API
│   │   ├── base.py          # AdapterMerger ABC
│   │   ├── linear.py        # Weighted average
│   │   ├── ties.py          # TIES (Trim, Elect Sign, Merge)
│   │   ├── dare.py          # DARE (Drop And REscale)
│   │   └── utils.py         # Weight loading utilities
│   ├── serving/
│   │   └── server.py        # FastAPI inference server
│   └── utils/
│       └── visualization.py # Routing analysis tools
├── tests/
├── examples/
│   └── config.yaml
├── pyproject.toml
└── README.md
```

## Development

```bash
git clone https://github.com/irfanalii/concadptr.git
cd concadptr
pip install -e ".[dev]"
pytest
```

## Roadmap

- [x] Core library architecture
- [x] 3 routing strategies (soft, top-k, X-LoRA)
- [x] Per-layer routing hooks (2-pass forward with LoRA delta weighting)
- [x] Adapter registry with compatibility validation
- [x] Router training pipeline
- [x] FastAPI serving
- [x] Routing visualization and analysis
- [x] Static merging — Linear, TIES, DARE, DARE+TIES
- [x] HuggingFace Hub push/pull (models and adapters)
- [x] Per-layer routing hooks in generation loop (cached prompt routing, KV-cache compatible)
- [x] Benchmarking suite — MMLU, HellaSwag, BLEU/ROUGE, A/B comparison, forgetting check
- [x] Adapter version metadata — version, training config hash, eval metrics, sidecar auto-load
- [ ] Progressive merging pipeline — incremental adapter integration with quality gating
- [ ] vLLM integration for high-throughput serving
- [ ] Federated LoRA training (FedAvg on adapter weights)

## References

- Hu et al. (2021) — [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- Zadouri et al. (2023) — [Pushing Mixture of Experts to the Limit (MoLoRA)](https://arxiv.org/abs/2309.05444)
- Yadav et al. (2023) — [TIES-Merging: Resolving Interference When Merging Models](https://arxiv.org/abs/2306.01708)
- Yu et al. (2023) — [Language Models are Super Mario (DARE)](https://arxiv.org/abs/2311.03099)
- Wu et al. (2024) — [Mixture of LoRA Experts (MoLE)](https://arxiv.org/abs/2404.13628)
- Li et al. (2024) — [MixLoRA](https://arxiv.org/abs/2404.15159)
- Buehler & Buehler (2024) — [X-LoRA: Mixture of Low-Rank Adapter Experts](https://github.com/EricLBuehler/xlora)

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Author

**Irfan Ali** — [GitHub](https://github.com/irfanalee) · [HuggingFace](https://huggingface.co/irfanalee)

---

*ConcAdptr — because the best models are concocted, not just trained.*
