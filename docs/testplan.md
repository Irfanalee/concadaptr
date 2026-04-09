# ConcAdptr — End-to-End Test Plan

Real-world validation of the full ConcAdptr pipeline: adapter training → concoction → routing → benchmarking → merging.

---

## Environment

```bash
git clone https://github.com/irfanalii/concadptr.git
cd concadptr
pip install -e ".[dev,benchmarks]"
```

**Hardware:** CPU is sufficient for the 0.5B smoke test. GPU (≥8GB VRAM) recommended for training steps.

---

## Model & Adapters

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-0.5B` |
| **LoRA rank** | 8 |
| **LoRA alpha** | 16 |
| **Target modules** | `q_proj`, `v_proj` |

Three domain adapters, each fine-tuned on a small public dataset:

| Name | Dataset |
|------|---------|
| `medical` | `medalpaca/medical_meadow_medical_flashcards` |
| `legal` | `nguyen-brat/legal_summarization` |
| `finance` | `gbharti/finance-alpaca` (first 2 000 examples) |

Train each adapter with PEFT + SFTTrainer, 3 epochs, saving to `./adapters/{name}/`.

---

## Test Scenarios

### 1. Adapter Registration & Compatibility

Register all three adapters and validate they are compatible for fusion.

```python
from concadptr.adapters import AdapterRegistry

registry = AdapterRegistry()
registry.register("medical", "./adapters/medical")
registry.register("legal",   "./adapters/legal")
registry.register("finance", "./adapters/finance")
registry.validate_compatibility()
print(registry.summary())
```

**Pass:** No exception raised, `registry.num_adapters == 3`, summary shows rank=8 for all.

---

### 2. Router Training

Build the concocted model and train the router on a mixed dataset.

```python
from concadptr import ConcAdptrConfig, ConcAdptrModel, ConcAdptrTrainer
from datasets import load_dataset, concatenate_datasets

config = ConcAdptrConfig(
    base_model="Qwen/Qwen2.5-0.5B",
    adapters={
        "medical": "./adapters/medical",
        "legal":   "./adapters/legal",
        "finance": "./adapters/finance",
    },
    routing_strategy="soft_merging",
    quantization=None,
)
model = ConcAdptrModel.from_config(config)

train_dataset = concatenate_datasets([
    load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train[:200]"),
    load_dataset("nguyen-brat/legal_summarization",             split="train[:200]"),
    load_dataset("gbharti/finance-alpaca",                      split="train[:200]"),
])

trainer = ConcAdptrTrainer(model=model, train_dataset=train_dataset,
                           learning_rate=1e-4, num_epochs=3, batch_size=4)
trainer.train()
model.save_pretrained("./concocted_model")
```

**Pass:** `train_loss` decreases across epochs; `lb_loss` stays below 0.5 (no expert collapse).

---

### 3. Routing in Inference

Verify that domain-specific prompts activate the expected adapter.

```python
from concadptr.utils import print_routing_summary

model.router.enable_history(True)

prompts = {
    "medical": "What are the symptoms of Type 2 diabetes?",
    "legal":   "Summarize the liability clauses in this contract.",
    "finance": "Explain the impact of rising interest rates on bond prices.",
}

for domain, prompt in prompts.items():
    inputs = tokenizer(prompt, return_tensors="pt")
    model(inputs["input_ids"])

stats = model.router.get_routing_stats()
print_routing_summary(stats, expert_names=["medical", "legal", "finance"])
model.router.enable_history(False)
```

**Pass:** The dominant expert (highest mean weight) for each prompt aligns with the expected domain adapter.

---

### 4. Routing in Generation

Run `model.generate()` and confirm routing hooks fire without errors.

```python
for domain, prompt in prompts.items():
    inputs = tokenizer(prompt, return_tensors="pt")
    output_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=64,
        do_sample=False,
    )
    print(f"[{domain}]", tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

**Pass:** All three prompts produce coherent text; no shape errors or hook exceptions.

---

### 5. Benchmark — A/B Comparison

Compare base model, each individual adapter, and full routing side by side.

```python
from concadptr import BenchmarkRunner, BenchmarkConfig

config = BenchmarkConfig(tasks=["mmlu", "hellaswag"], num_samples=50)
runner = BenchmarkRunner(model, config)
comparison = runner.compare()

for variant, results in comparison.items():
    for r in results:
        print(f"{variant:20s} {r.task:12s} accuracy={r.metrics.get('accuracy', 0):.3f}")
```

**Pass:** Routed model accuracy ≥ base model on both tasks; ideally ≥ best single adapter on at least one task.

---

### 6. Forgetting Check

Confirm the concocted model does not catastrophically forget general capabilities.

```python
deltas = runner.forgetting_check()
for task, d in deltas.items():
    print(f"{task}: base={d['base']:.3f}  routed={d['routed']:.3f}  delta={d['delta']:+.3f}")
```

**Pass:** Both MMLU and HellaSwag deltas > −0.05 (less than 5% accuracy drop vs base model).

---

### 7. Adapter Version Metadata

Attach provenance metadata to an adapter and verify it round-trips through the registry.

```python
from concadptr.adapters import AdapterInfo

training_config = {"learning_rate": 1e-4, "num_epochs": 3, "rank": 8}
config_hash = AdapterInfo.compute_config_hash(training_config)

info = registry.get("medical")
info.version = "1.0.0"
info.training_config_hash = config_hash
info.eval_metrics = {"mmlu": 0.61}
info.save_version_metadata()

# Re-register from disk and confirm fields load automatically
registry2 = AdapterRegistry()
info2 = registry2.register("medical", "./adapters/medical")
assert info2.version == "1.0.0"
assert info2.training_config_hash == config_hash
```

**Pass:** `concadptr_version.json` written to `./adapters/medical/`; fields restored on re-registration.

---

### 8. Static Merge — TIES

Merge all three adapters into a single PEFT adapter and verify it is loadable.

```python
from concadptr import merge_adapters
from peft import PeftModel
from transformers import AutoModelForCausalLM

output_path = merge_adapters(
    adapters={
        "medical": "./adapters/medical",
        "legal":   "./adapters/legal",
        "finance": "./adapters/finance",
    },
    output_path="./merged_ties",
    method="ties",
    trim_fraction=0.2,
)

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
merged_model = PeftModel.from_pretrained(base, str(output_path))
print("Merged adapter loaded successfully.")
```

**Pass:** No exception; merged adapter produces coherent output on a sample prompt.

---

### 9. Progressive Merging Pipeline

Start with 2 adapters, then incrementally add the third with quality gating.

```python
from concadptr import ConcAdptrConfig, ConcAdptrModel
from concadptr.merging.progressive import ProgressiveMerger, ProgressiveMergerConfig

config_2 = ConcAdptrConfig(
    base_model="Qwen/Qwen2.5-0.5B",
    adapters={"medical": "./adapters/medical", "legal": "./adapters/legal"},
    routing_strategy="soft_merging",
    quantization=None,
)
model_2 = ConcAdptrModel.from_config(config_2)

merger = ProgressiveMerger(model_2, ProgressiveMergerConfig(
    quality_gate_threshold=-0.05,
    merge_method="ties",
    benchmark_num_samples=50,
))

result = merger.add_adapter("finance", "./adapters/finance", "./merged_progressive")
print(f"Gate passed: {result.passed_gate}")
print(f"Deltas: {result.deltas}")
print(f"Merged to: {result.output_path}")
```

**Pass:** `result.passed_gate == True`; merged adapter written to `./merged_progressive/`.

---

### 10. HuggingFace Hub Round-trip *(optional)*

Push an adapter to the Hub and pull it back.

```python
registry.push_adapter_to_hub("medical", repo_id="your-username/concadptr-medical-test")

registry2 = AdapterRegistry()
info = registry2.load_adapter_from_hub("your-username/concadptr-medical-test", name="medical")
assert info.rank == 8
```

**Pass:** Adapter uploaded and re-registered with matching rank and base model; no errors.

---

## Expected Results Summary

| # | Scenario | Pass Criteria |
|---|----------|---------------|
| 1 | Registration | No exception, 3 adapters, rank=8 all |
| 2 | Router training | Loss decreases, lb_loss < 0.5 |
| 3 | Routing in inference | Dominant expert matches domain |
| 4 | Routing in generation | Coherent text, no errors |
| 5 | A/B comparison | Routed ≥ base on both tasks |
| 6 | Forgetting check | Deltas > −0.05 |
| 7 | Version metadata | Fields round-trip through JSON sidecar |
| 8 | Static merge (TIES) | Merged adapter loads and generates |
| 9 | Progressive merger | Gate passes, merged adapter written |
| 10 | Hub round-trip | Adapter re-registered with correct metadata |
