"""
FastAPI inference server for ConcAdptr models.

Provides OpenAI-compatible /v1/completions endpoint with
automatic expert routing.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def create_app(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    """Create a FastAPI app for serving a ConcAdptr model.

    Args:
        model_path: Path to saved ConcAdptr model directory.
        host: Server host.
        port: Server port.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError:
        raise ImportError(
            "Serving dependencies not installed. "
            "Install with: pip install concadptr[serving]"
        )

    app = FastAPI(
        title="ConcAdptr Inference Server",
        description="Mixture of LoRA Experts inference with learned routing.",
        version="0.1.0",
    )

    # Model will be loaded on startup
    _model_state = {"model": None}

    @app.on_event("startup")
    async def load_model():
        from concadptr.model import ConcAdptrModel

        logger.info(f"Loading ConcAdptr model from {model_path}...")
        _model_state["model"] = ConcAdptrModel.load_pretrained(model_path)
        _model_state["model"].router.eval()
        logger.info("Model loaded and ready for inference.")

    class CompletionRequest(BaseModel):
        prompt: str
        max_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.9
        adapter: Optional[str] = None  # Force specific adapter (bypass router)

    class CompletionResponse(BaseModel):
        text: str
        routing_weights: dict
        tokens_generated: int

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        model = _model_state["model"]
        tokenizer = model.tokenizer

        inputs = tokenizer(request.prompt, return_tensors="pt")
        device = next(model.router.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate with routing
        # TODO: Implement proper generation loop with routing
        # For now, return a placeholder
        return CompletionResponse(
            text="[Generation not yet implemented — router training works!]",
            routing_weights={
                name: 1.0 / model.registry.num_adapters
                for name in model.registry.names
            },
            tokens_generated=0,
        )

    @app.get("/v1/adapters")
    async def list_adapters():
        model = _model_state["model"]
        return {
            "adapters": model.registry.names,
            "num_adapters": model.registry.num_adapters,
            "routing_strategy": model.config.router.strategy.value,
        }

    @app.get("/health")
    async def health():
        return {"status": "ok", "model_loaded": _model_state["model"] is not None}

    return app


def serve(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Start the ConcAdptr inference server.

    Args:
        model_path: Path to saved ConcAdptr model.
        host: Server host.
        port: Server port.
    """
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "Serving dependencies not installed. "
            "Install with: pip install concadptr[serving]"
        )

    app = create_app(model_path, host, port)
    uvicorn.run(app, host=host, port=port)
