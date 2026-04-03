"""
ConcAdptrTrainer — Training loop for the ConcAdptr router.

Trains only the routing/gating network while keeping the base model
and all LoRA adapters frozen. Designed for efficiency on consumer GPUs.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ConcAdptrTrainer:
    """Trainer for ConcAdptr router networks.

    Only trains the router parameters. Base model and adapter weights
    remain frozen throughout training.

    Example:
        >>> trainer = ConcAdptrTrainer(model, train_dataset, eval_dataset)
        >>> trainer.train()
        >>> model.save_pretrained("./output")

    Args:
        model: ConcAdptrModel instance.
        train_dataset: Training dataset (HuggingFace Dataset or compatible).
        eval_dataset: Optional evaluation dataset.
        output_dir: Directory for saving checkpoints.
        learning_rate: Learning rate for router training.
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        max_seq_length: Maximum sequence length.
        warmup_ratio: Warmup proportion of total steps.
        weight_decay: Weight decay coefficient.
        gradient_accumulation_steps: Gradient accumulation steps.
        fp16: Use automatic mixed precision.
        logging_steps: Log every N steps.
        eval_steps: Evaluate every N steps.
        save_steps: Save checkpoint every N steps.
    """

    def __init__(
        self,
        model: Any,  # ConcAdptrModel — avoid circular import
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = "./concadptr_output",
        learning_rate: float = 1e-4,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_seq_length: int = 2048,
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        fp16: bool = True,
        logging_steps: int = 10,
        eval_steps: int = 50,
        save_steps: int = 100,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps

        # Training state
        self.global_step = 0
        self.best_eval_loss = float("inf")
        self._training_log: list = []

    def _create_dataloader(self, dataset: Any, shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from the dataset.

        Args:
            dataset: HuggingFace Dataset or PyTorch Dataset.
            shuffle: Whether to shuffle the data.

        Returns:
            DataLoader instance.
        """
        tokenizer = self.model.tokenizer

        def collate_fn(examples):
            # Handle both dict-style and text-style datasets
            if isinstance(examples[0], dict):
                texts = [ex.get("text", ex.get("input", "")) for ex in examples]
            elif isinstance(examples[0], str):
                texts = examples
            else:
                texts = [str(ex) for ex in examples]

            encoded = tokenizer(
                texts,
                max_length=self.max_seq_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encoded["labels"] = encoded["input_ids"].clone()
            return encoded

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
        )

    def train(self) -> Dict[str, Any]:
        """Run the full training loop for the router.

        Returns:
            Training statistics dictionary.
        """
        logger.info("=" * 60)
        logger.info("ConcAdptr Router Training")
        logger.info("=" * 60)
        logger.info(f"  Adapters: {self.model.registry.names}")
        logger.info(f"  Router: {self.model.config.router.strategy.value}")
        logger.info(f"  Trainable params: {self.model.get_num_trainable_params():,}")
        logger.info(f"  Epochs: {self.num_epochs}")
        logger.info(f"  Batch size: {self.batch_size}")
        logger.info(f"  Learning rate: {self.learning_rate}")
        logger.info(f"  Output dir: {self.output_dir}")
        logger.info("=" * 60)

        # Create dataloaders
        train_loader = self._create_dataloader(self.train_dataset, shuffle=True)
        eval_loader = None
        if self.eval_dataset:
            eval_loader = self._create_dataloader(self.eval_dataset, shuffle=False)

        # Optimizer — only router parameters
        optimizer = AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        total_steps = len(train_loader) * self.num_epochs // self.gradient_accumulation_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.fp16 else None

        # Training loop
        self.model.router.train()
        start_time = time.time()

        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            epoch_lm_loss = 0.0
            epoch_lb_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{self.num_epochs}",
                total=len(train_loader),
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                device = next(self.model.router.parameters()).device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Forward pass
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)

                loss = outputs["loss"]
                if loss is None:
                    continue

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                # Optimizer step
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1

                # Track losses
                epoch_loss += outputs["loss"].item() if outputs["loss"] is not None else 0
                epoch_lm_loss += outputs["lm_loss"].item() if outputs["lm_loss"] is not None else 0
                epoch_lb_loss += outputs["load_balance_loss"].item()
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{epoch_loss / num_batches:.4f}",
                    "lm": f"{epoch_lm_loss / num_batches:.4f}",
                    "lb": f"{epoch_lb_loss / num_batches:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # Logging
                if self.global_step % self.logging_steps == 0 and self.global_step > 0:
                    log_entry = {
                        "step": self.global_step,
                        "epoch": epoch + 1,
                        "loss": epoch_loss / num_batches,
                        "lm_loss": epoch_lm_loss / num_batches,
                        "load_balance_loss": epoch_lb_loss / num_batches,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    self._training_log.append(log_entry)

                # Evaluation
                if (
                    eval_loader
                    and self.global_step % self.eval_steps == 0
                    and self.global_step > 0
                ):
                    eval_results = self._evaluate(eval_loader)
                    logger.info(
                        f"Step {self.global_step} | "
                        f"Eval Loss: {eval_results['eval_loss']:.4f}"
                    )

                    # Save best model
                    if eval_results["eval_loss"] < self.best_eval_loss:
                        self.best_eval_loss = eval_results["eval_loss"]
                        self.model.save_pretrained(self.output_dir / "best")
                        logger.info(f"  New best model saved (loss: {self.best_eval_loss:.4f})")

                # Save checkpoint
                if self.global_step % self.save_steps == 0 and self.global_step > 0:
                    checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
                    self.model.save_pretrained(checkpoint_dir)

            # End of epoch summary
            avg_loss = epoch_loss / max(num_batches, 1)
            logger.info(
                f"Epoch {epoch + 1} complete | "
                f"Avg Loss: {avg_loss:.4f} | "
                f"LM Loss: {epoch_lm_loss / max(num_batches, 1):.4f} | "
                f"LB Loss: {epoch_lb_loss / max(num_batches, 1):.4f}"
            )

        # Training complete
        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.1f}s ({elapsed / 60:.1f}m)")

        # Save final model
        self.model.save_pretrained(self.output_dir / "final")

        return {
            "total_steps": self.global_step,
            "training_time_seconds": elapsed,
            "best_eval_loss": self.best_eval_loss,
            "final_loss": avg_loss,
            "training_log": self._training_log,
        }

    @torch.no_grad()
    def _evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """Run evaluation on the eval dataset.

        Args:
            eval_loader: Evaluation DataLoader.

        Returns:
            Dictionary with evaluation metrics.
        """
        self.model.router.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_lb_loss = 0.0
        num_batches = 0

        for batch in eval_loader:
            device = next(self.model.router.parameters()).device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = self.model(**batch)
            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                total_lm_loss += outputs["lm_loss"].item() if outputs["lm_loss"] is not None else 0
                total_lb_loss += outputs["load_balance_loss"].item()
                num_batches += 1

        self.model.router.train()

        n = max(num_batches, 1)
        return {
            "eval_loss": total_loss / n,
            "eval_lm_loss": total_lm_loss / n,
            "eval_lb_loss": total_lb_loss / n,
        }
