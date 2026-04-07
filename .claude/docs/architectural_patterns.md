# ConcAdptr — Architectural Patterns

Patterns that appear in **two or more source files** in `concadptr/`. Use this as the
reference when adding new modules or reviewing PRs.

---

## 1. `from __future__ import annotations` Header

**Files:** every module (config.py, model.py, trainer.py, router/*, adapters/__init__.py, serving/server.py, utils/visualization.py)

```python
from __future__ import annotations
```

Enables deferred evaluation of type hints. Allows forward references (`"ConcAdptrModel"`)
and keeps annotations as strings at runtime — zero overhead, works on Python 3.9+.
**Always the first non-comment line in every module.**

---

## 2. Google-Style Docstrings

**Files:** all public classes and methods across the entire package

```python
def register(self, name: str, path: Union[str, Path]) -> AdapterInfo:
    """Register a LoRA adapter with the registry.

    Args:
        name: Unique name for this adapter.
        path: Path to the adapter directory.

    Returns:
        AdapterInfo with parsed adapter details.

    Raises:
        FileNotFoundError: If the adapter path doesn't exist.
        ValueError: If the adapter name is already registered.
    """
```

All public methods document Args, Returns, and Raises where applicable. Private
methods (leading `_`) may omit docstrings.

---

## 3. `Union[str, Path]` Input + `Path(path)` Normalisation

**Files:** `concadptr/config.py`, `concadptr/model.py`, `concadptr/trainer.py`

```python
# Accept either — normalise immediately
def save(self, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ...

def save_pretrained(self, path: Union[str, Path]) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ...
```

Public methods accept `str | Path` for caller convenience. The first line always
converts to `Path` so the rest of the method uses chainable `pathlib` operations.
Never use `os.path` — `pathlib` only.

---

## 4. Dataclass Config with `__post_init__` for Derived State

**Files:** `concadptr/config.py` (RouterConfig, TrainingConfig, ServingConfig, ConcAdptrConfig), `concadptr/adapters/__init__.py` (AdapterInfo)

```python
@dataclass
class ConcAdptrConfig:
    routing_strategy: str = "xlora"
    router: RouterConfig = field(default_factory=RouterConfig)

    def __post_init__(self):
        # Sync shorthand string → typed enum on the nested config
        if self.routing_strategy:
            self.router.strategy = RoutingStrategy(self.routing_strategy)
```

Config objects are plain dataclasses — no Pydantic, no extra deps. Mutable defaults
use `field(default_factory=...)`. `__post_init__` handles any state that must be
derived from other fields after construction.

---

## 5. `str` Enum for Strategy Selection + Model Dispatch

**Files:** `concadptr/config.py` (definition), `concadptr/model.py` (dispatch), `concadptr/trainer.py` + `concadptr/serving/server.py` (`.value` access)

```python
# config.py
class RoutingStrategy(str, Enum):
    SOFT_MERGING = "soft_merging"
    TOP_K        = "top_k"
    XLORA        = "xlora"

# model.py — dispatch on enum
strategy = self.config.router.strategy
if strategy == RoutingStrategy.SOFT_MERGING:
    self.router = SoftMergingRouter(...)
elif strategy == RoutingStrategy.TOP_K:
    self.router = TopKRouter(...)
elif strategy == RoutingStrategy.XLORA:
    self.router = XLoRARouter(...)

# trainer.py, server.py — human-readable string
logger.info(f"Router: {self.model.config.router.strategy.value}")
```

`str` enum means the value is both type-safe in Python and serialises directly to
YAML/JSON without a custom encoder. Always compare against the enum constant, never
the raw string.

---

## 6. Abstract Base Class + `nn.Module` Dual Inheritance

**Files:** `concadptr/router/base.py` (ABC), `concadptr/router/soft_merging.py`, `top_k.py`, `xlora.py` (concrete)

```python
# base.py
class BaseRouter(ABC, nn.Module):
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor,
                layer_idx: Optional[int] = None) -> torch.Tensor: ...

    # Shared concrete methods all subclasses inherit:
    def compute_load_balance_loss(self, routing_weights): ...
    def enable_history(self, enable: bool = True): ...
    def get_routing_stats(self) -> Dict[str, torch.Tensor]: ...

# Concrete subclass
class SoftMergingRouter(BaseRouter):
    def forward(self, hidden_states, layer_idx=None) -> torch.Tensor:
        ...  # must implement
```

ABC enforces the `forward()` contract at import time. `nn.Module` gives PyTorch
parameter tracking for free. Shared behaviour (load-balance loss, history recording)
lives in `BaseRouter` — never duplicated in subclasses.

---

## 7. `nn.ModuleList` for Per-Layer Gate Parameters

**Files:** `concadptr/router/soft_merging.py`, `concadptr/router/top_k.py`, `concadptr/router/xlora.py`

```python
if use_layerwise:
    self.gates = nn.ModuleList([
        nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_experts),
        )
        for _ in range(num_layers)
    ])
else:
    self.gate = nn.Sequential(...)   # shared across all layers
```

`nn.ModuleList` ensures per-layer parameters are registered with PyTorch and appear
in `state_dict()`. The `use_layerwise` flag is a constructor argument on all three
routers — always check it before accessing `self.gates[layer_idx]`.

---

## 8. `@classmethod` Factory Constructors

**Files:** `concadptr/config.py` (`from_yaml`), `concadptr/model.py` (`from_config`, `from_yaml`, `load_pretrained`)

```python
# Named alternative constructors — preferred over overloading __init__
@classmethod
def from_config(cls, config: ConcAdptrConfig) -> "ConcAdptrModel":
    issues = config.validate()
    errors = [i for i in issues if i.startswith("ERROR")]
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(errors))
    model = cls(config)          # __init__ stays simple
    model._load_base_model()
    model._load_adapters()
    model._init_router()
    return model

@classmethod
def from_yaml(cls, path: Union[str, Path]) -> "ConcAdptrModel":
    return cls.from_config(ConcAdptrConfig.from_yaml(path))
```

`__init__` only sets attributes. All complex initialisation (I/O, validation,
model loading) goes in a named classmethod. This makes the construction path
explicit and individually testable.

---

## 9. Paired `save_pretrained` / `load_pretrained` Checkpoint Methods

**Files:** `concadptr/model.py` (definition), `concadptr/trainer.py` (call sites)

```python
# model.py
def save_pretrained(self, path: Union[str, Path]) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(self.router.state_dict(), path / "router.pt")
    self.config.save(path / "concadptr_config.yaml")
    # adapter registry metadata as JSON

@classmethod
def load_pretrained(cls, path: Union[str, Path]) -> "ConcAdptrModel":
    config = ConcAdptrConfig.from_yaml(path / "concadptr_config.yaml")
    model = cls.from_config(config)
    router_state = torch.load(path / "router.pt", map_location="cpu")
    model.router.load_state_dict(router_state)
    return model

# trainer.py — three call sites
self.model.save_pretrained(self.output_dir / "best")
self.model.save_pretrained(self.output_dir / f"checkpoint-{self.global_step}")
self.model.save_pretrained(self.output_dir / "final")
```

Only the **router weights** and **config** are saved — not the base model or
adapters (those are referenced by path). `map_location="cpu"` on load keeps it
device-agnostic; the caller moves to GPU if needed.

---

## 10. Module-Level Logger via `__name__`

**Files:** `concadptr/model.py`, `concadptr/trainer.py`, `concadptr/adapters/__init__.py`, `concadptr/serving/server.py`

```python
import logging
logger = logging.getLogger(__name__)

# Usage throughout the module:
logger.info(f"Loading base model: {self.config.base_model}")
logger.warning(f"Adapter path not found: {path} ({name}), skipping")
```

One logger per module, named after the module. Callers configure logging at the
application level — the library never calls `logging.basicConfig()`. This follows
the standard Python library convention.

---

## 11. Lazy Optional-Dependency Imports with Install Hint

**Files:** `concadptr/serving/server.py`, `concadptr/utils/visualization.py`

```python
# serving/server.py — inside create_app()
try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "Serving dependencies not installed. "
        "Install with: pip install concadptr[serving]"
    )

# utils/visualization.py — inside each plot function
try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    raise ImportError(
        "Visualization requires matplotlib. "
        "Install with: pip install matplotlib"
    )
```

Optional dependencies (`fastapi`, `uvicorn`, `matplotlib`) are imported **inside
the function** that needs them, not at module top-level. This keeps `import concadptr`
fast and dependency-free for users who only need the core training path.

---

## 12. Explicit Parameter Freezing

**Files:** `concadptr/config.py` (flags), `concadptr/model.py` (enforcement), `concadptr/trainer.py` (optimizer scope)

```python
# config.py — opt-out flags
freeze_adapters: bool = True
freeze_base_model: bool = True

# model.py — freeze base model
if self.config.freeze_base_model:
    for param in self.base_model.parameters():
        param.requires_grad = False

# model.py — freeze adapter LoRA weights by name
if self.config.freeze_adapters:
    for name, param in self.base_model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = False

# model.py — context manager guard during forward
with torch.no_grad() if self.config.freeze_adapters else torch.enable_grad():
    outputs = self.base_model(...)

# trainer.py — optimizer only sees router params
optimizer = AdamW(self.model.get_trainable_parameters(), ...)
```

Freezing is applied at three independent layers (requires_grad flag, context manager,
optimizer scope) so a bug in any one layer can't accidentally leak gradients into
frozen weights.

---

## 13. `validate()` Returns Issue List, Doesn't Raise

**Files:** `concadptr/config.py`, consumed in `concadptr/model.py`

```python
# config.py — accumulate, don't raise
def validate(self) -> List[str]:
    issues = []
    if not self.base_model:
        issues.append("ERROR: base_model is required")
    if len(self.adapters) < 2:
        issues.append("WARNING: ConcAdptr works best with 2+ adapters.")
    ...
    return issues  # caller decides what to do

# model.py — consumer separates errors from warnings
issues = config.validate()
errors = [i for i in issues if i.startswith("ERROR")]
if errors:
    raise ValueError("Configuration errors:\n" + "\n".join(errors))
for warning in [i for i in issues if i.startswith("WARNING")]:
    logger.warning(warning)
```

`validate()` collects **all** problems before stopping. Errors are prefixed `"ERROR:"`,
warnings `"WARNING:"`. The caller decides whether to raise or log. This lets tools
surface every problem at once rather than failing on the first one.

---

## 14. Dict Return Values from Complex Operations

**Files:** `concadptr/model.py` (`forward()`), `concadptr/trainer.py` (`train()`, `_evaluate()`)

```python
# model.py
return {
    "loss": total_loss,
    "lm_loss": loss,
    "logits": fused_logits,
    "routing_weights": routing_weights.detach(),
    "load_balance_loss": lb_loss.detach(),
}

# trainer.py
return {
    "total_steps": self.global_step,
    "training_time_seconds": elapsed,
    "best_eval_loss": self.best_eval_loss,
    "final_loss": avg_loss,
    "training_log": self._training_log,
}
```

Structured dicts over tuples — keys are self-documenting, callers can destructure
only what they need, and adding new fields is non-breaking. Training loss keys
(`loss`, `lm_loss`, `load_balance_loss`) are stable across all call sites.

---

## 16. Two-Pass Forward with Persistent Hook Infrastructure

**Files:** `concadptr/model.py` (`forward()`, `generate()`, `_register_lora_hooks()`, `_make_lora_hook()`, `_make_generation_hook()`)

```python
# Pass 1: base model only → per-layer hidden states
with self.base_model.disable_adapter():
    base_out = self.base_model(..., output_hidden_states=True)

# Compute routing weights per layer
layer_weights = [self.router(h.detach(), layer_idx=i) for i, h in enumerate(layer_hiddens)]
routing_weights_all = torch.stack(layer_weights, dim=2)  # (batch, seq, num_layers, num_experts)

# Pass 2: single forward with routing hooks active
hooks = self._register_lora_hooks(routing_weights_all, self._make_lora_hook)
try:
    fwd_out = self.base_model(...)
finally:
    for h in hooks:
        h.remove()
```

`_register_lora_hooks(weights, hook_factory)` centralises hook registration — used by both `forward()` and `generate()`. The `hook_factory` argument is either `_make_lora_hook` (training/forward, full seq weights) or `_make_generation_hook` (generation, cached last-token weights). Hooks are always removed in a `finally` block — even if an exception occurs.

---

## 17. Cached Prompt Routing for Generation

**Files:** `concadptr/model.py` (`generate()`, `_make_generation_hook()`)

```python
@torch.no_grad()
def generate(self, input_ids, attention_mask=None, **kwargs):
    # Pass 1 on full prompt → routing weights
    cached_routing = routing_weights_all[:, -1:, :, :]  # (batch, 1, num_layers, num_experts)

    # Persistent hooks with cached weights — KV-cache compatible
    hooks = self._register_lora_hooks(cached_routing, self._make_generation_hook)
    try:
        return self.base_model.generate(...)
    finally:
        for h in hooks:
            h.remove()
```

`_make_generation_hook` receives `cached_weights` of shape `(batch, 1, num_layers, num_experts)`. Inside the hook, `weights = cached_weights[:, :, layer_idx, :]` is `(batch, 1, num_experts)` which broadcasts safely over any generation step input of shape `(batch, seq_step, hidden)` — including KV-cache steps where `seq_step=1`. The last token's routing weights serve as the routing proxy for all generated tokens.

---

## 15. `_private` Attributes for Internal State

**Files:** `concadptr/model.py`, `concadptr/router/base.py`, `concadptr/trainer.py`

```python
# model.py
self._adapter_models: Dict[str, nn.Module] = {}
self._is_loaded = False

# router/base.py
self._routing_history: list = []
self._record_history = False

# trainer.py
self._training_log: list = []
```

A single leading underscore signals "implementation detail — do not depend on this
from outside the class." Public API surfaces (`registry`, `router`, `config`,
`global_step`) have no underscore. This convention is enforced by code review, not
by Python's name mangling.
