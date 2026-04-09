"""
Benchmark task implementations for ConcAdptr.

Each task loads a HuggingFace dataset, runs inference, and returns a BenchmarkResult.

Multiple-choice tasks (MMLUTask, HellaSwagTask) use log-probability scoring rather
than generation: for each candidate answer, compute the sum of log P(token | prefix)
over the answer tokens, then pick the argmax. This requires only a forward pass, not
autoregressive generation, which keeps evaluation fast.

GenerationTask runs model.generate() and computes BLEU/ROUGE against references.
Requires: pip install 'concadptr[benchmarks]'
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from concadptr.benchmarks.config import BenchmarkConfig, BenchmarkResult
from concadptr.benchmarks.metrics import accuracy, bleu, rouge

if TYPE_CHECKING:
    from concadptr.model import ConcAdptrModel


def _get_logits(
    model: ConcAdptrModel,
    input_ids: torch.Tensor,
    adapter_name: str,
) -> torch.Tensor:
    """Run a forward pass and return the logits tensor (batch, seq, vocab).

    Handles three modes based on adapter_name:
    - "concadptr_routed": full routing via ConcAdptrModel.forward()
    - "base": base model with all adapters disabled
    - any other string: base model with that adapter active (single-adapter mode)

    Args:
        model: Initialized ConcAdptrModel.
        input_ids: Token IDs of shape (1, seq_len).
        adapter_name: Which model variant to use.

    Returns:
        Logits tensor of shape (1, seq_len, vocab_size).
    """
    if adapter_name == "concadptr_routed":
        return model(input_ids)["logits"]
    elif adapter_name == "base":
        with model.base_model.disable_adapter():
            return model.base_model(input_ids).logits
    else:
        model.base_model.set_adapter(adapter_name)
        return model.base_model(input_ids).logits


def _score_choices(
    model: ConcAdptrModel,
    tokenizer,
    prefix: str,
    choices: list[str],
    device: torch.device,
    adapter_name: str,
    max_length: int = 512,
) -> int:
    """Score multiple-choice options via log-probability and return the argmax.

    For each choice, tokenizes (prefix + choice) and computes the sum of
    log P(choice_token_i | prefix + choice_tokens_0..i-1) over all choice tokens.
    Returns the index of the highest-scoring choice.

    Args:
        model: Initialized ConcAdptrModel.
        tokenizer: HuggingFace tokenizer.
        prefix: Question or context text (everything before the answer).
        choices: List of candidate answer strings.
        device: Device to move input tensors to.
        adapter_name: Model variant to use (see _get_logits).
        max_length: Maximum token length; longer sequences are truncated.

    Returns:
        Index of the highest-scoring choice.
    """
    prefix_ids = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]
    prefix_len = max(1, prefix_ids.shape[1])

    scores = []
    for choice in choices:
        full_ids = tokenizer(
            prefix + choice,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
        )["input_ids"].to(device)

        # Clamp prefix_len so there is always at least one choice token to score.
        actual_prefix_len = min(prefix_len, full_ids.shape[1] - 1)

        with torch.no_grad():
            logits = _get_logits(model, full_ids, adapter_name)  # (1, seq_len, vocab)

        # Score: sum log P(token_t | tokens_0..t-1) for t in [actual_prefix_len, seq_len-1]
        # logits[0, t-1, :] is the distribution over token_t.
        log_probs = torch.log_softmax(
            logits[0, actual_prefix_len - 1 : -1, :], dim=-1
        )  # (num_choice_tokens, vocab)
        choice_token_ids = full_ids[0, actual_prefix_len:]  # (num_choice_tokens,)

        if len(choice_token_ids) == 0:
            scores.append(-float("inf"))
            continue

        score = log_probs[torch.arange(len(choice_token_ids)), choice_token_ids].sum().item()
        scores.append(score)

    return int(scores.index(max(scores)))


class BenchmarkTask(ABC):
    """Abstract base class for ConcAdptr benchmark tasks.

    Subclasses override `name` and implement `evaluate()` for a specific dataset.
    """

    name: str  # Override in each subclass; used for registration and result labels.

    @abstractmethod
    def evaluate(
        self,
        model: ConcAdptrModel,
        config: BenchmarkConfig,
        adapter_name: str = "concadptr_routed",
    ) -> BenchmarkResult:
        """Run the benchmark and return a result.

        Args:
            model: Initialized ConcAdptrModel.
            config: Benchmark configuration.
            adapter_name: Model variant to evaluate. "concadptr_routed" uses full
                routing; "base" disables all adapters; any other string selects
                that named adapter.

        Returns:
            BenchmarkResult with task metrics and timing.
        """
        ...


class MMLUTask(BenchmarkTask):
    """MMLU (Massive Multitask Language Understanding) benchmark.

    Evaluates accuracy on 57-subject multiple-choice questions using log-probability
    scoring. No generation is required.

    Dataset: cais/mmlu (HuggingFace Hub)
    Metric: accuracy
    """

    name = "mmlu"

    def evaluate(
        self,
        model: ConcAdptrModel,
        config: BenchmarkConfig,
        adapter_name: str = "concadptr_routed",
    ) -> BenchmarkResult:
        """Evaluate MMLU accuracy via log-prob scoring.

        Args:
            model: Initialized ConcAdptrModel.
            config: Benchmark configuration.
            adapter_name: Model variant to evaluate.

        Returns:
            BenchmarkResult with {"accuracy": float} in metrics.
        """
        import datasets as hf_datasets

        device = next(model.router.parameters()).device
        tokenizer = model.tokenizer

        dataset = hf_datasets.load_dataset("cais/mmlu", "all", split=config.mmlu_split)

        if config.mmlu_subjects:
            dataset = dataset.filter(lambda x: x["subject"] in config.mmlu_subjects)

        if config.num_samples is not None:
            dataset = dataset.select(range(min(config.num_samples, len(dataset))))

        preds: list[int] = []
        labels: list[int] = []
        start = time.time()

        for example in dataset:
            question: str = example["question"]
            choices: list[str] = example["choices"]
            label: int = int(example["answer"])

            pred = _score_choices(
                model, tokenizer, f"{question}\nAnswer: ", choices, device, adapter_name
            )
            preds.append(pred)
            labels.append(label)

        return BenchmarkResult(
            task=self.name,
            adapter_name=adapter_name,
            metrics={"accuracy": accuracy(preds, labels)},
            num_samples=len(labels),
            elapsed_seconds=time.time() - start,
        )


class HellaSwagTask(BenchmarkTask):
    """HellaSwag commonsense NLI benchmark.

    Evaluates accuracy on sentence-completion with 4 endings using log-probability
    scoring. No generation is required.

    Dataset: hellaswag (HuggingFace Hub)
    Metric: accuracy
    """

    name = "hellaswag"

    def evaluate(
        self,
        model: ConcAdptrModel,
        config: BenchmarkConfig,
        adapter_name: str = "concadptr_routed",
    ) -> BenchmarkResult:
        """Evaluate HellaSwag accuracy via log-prob scoring.

        Args:
            model: Initialized ConcAdptrModel.
            config: Benchmark configuration.
            adapter_name: Model variant to evaluate.

        Returns:
            BenchmarkResult with {"accuracy": float} in metrics.
        """
        import datasets as hf_datasets

        device = next(model.router.parameters()).device
        tokenizer = model.tokenizer

        dataset = hf_datasets.load_dataset("hellaswag", split=config.hellaswag_split)

        if config.num_samples is not None:
            dataset = dataset.select(range(min(config.num_samples, len(dataset))))

        preds: list[int] = []
        labels: list[int] = []
        start = time.time()

        for example in dataset:
            ctx: str = example["ctx"]
            endings: list[str] = example["endings"]
            label: int = int(example["label"])

            pred = _score_choices(
                model, tokenizer, ctx + " ", endings, device, adapter_name
            )
            preds.append(pred)
            labels.append(label)

        return BenchmarkResult(
            task=self.name,
            adapter_name=adapter_name,
            metrics={"accuracy": accuracy(preds, labels)},
            num_samples=len(labels),
            elapsed_seconds=time.time() - start,
        )


class GenerationTask(BenchmarkTask):
    """Custom generation task with BLEU/ROUGE evaluation.

    Loads a user-specified HuggingFace dataset, generates text with the model,
    and evaluates against reference strings.

    Dataset: Configured via config.generation_dataset.
    Metrics: bleu, rouge1, rouge2, rougeL (requires evaluate library).
    """

    name = "generation"

    def evaluate(
        self,
        model: ConcAdptrModel,
        config: BenchmarkConfig,
        adapter_name: str = "concadptr_routed",
    ) -> BenchmarkResult:
        """Generate text and compute BLEU/ROUGE against references.

        Args:
            model: Initialized ConcAdptrModel.
            config: Benchmark configuration. config.generation_dataset must be set.
            adapter_name: Model variant to evaluate.

        Returns:
            BenchmarkResult with metric keys from config.generation_metrics.

        Raises:
            ValueError: If config.generation_dataset is not set.
        """
        import datasets as hf_datasets

        if not config.generation_dataset:
            raise ValueError(
                "config.generation_dataset must be set to use GenerationTask."
            )

        device = next(model.router.parameters()).device
        tokenizer = model.tokenizer

        dataset = hf_datasets.load_dataset(
            config.generation_dataset, split=config.generation_split
        )

        if config.num_samples is not None:
            dataset = dataset.select(range(min(config.num_samples, len(dataset))))

        predictions: list[str] = []
        references: list[str] = []
        start = time.time()

        for example in dataset:
            input_text: str = example[config.generation_input_field]
            reference: str = example[config.generation_reference_field]

            input_ids = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )["input_ids"].to(device)

            with torch.no_grad():
                if adapter_name == "concadptr_routed":
                    output_ids = model.generate(
                        input_ids, max_new_tokens=config.max_new_tokens
                    )
                elif adapter_name == "base":
                    with model.base_model.disable_adapter():
                        output_ids = model.base_model.generate(
                            input_ids, max_new_tokens=config.max_new_tokens
                        )
                else:
                    model.base_model.set_adapter(adapter_name)
                    output_ids = model.base_model.generate(
                        input_ids, max_new_tokens=config.max_new_tokens
                    )

            new_tokens = output_ids[0, input_ids.shape[1] :]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)

        metrics: dict[str, float] = {}
        if "bleu" in config.generation_metrics:
            metrics["bleu"] = bleu(predictions, references)
        if "rouge" in config.generation_metrics:
            metrics.update(rouge(predictions, references))

        return BenchmarkResult(
            task=self.name,
            adapter_name=adapter_name,
            metrics=metrics,
            num_samples=len(predictions),
            elapsed_seconds=time.time() - start,
        )


_TASK_REGISTRY: dict[str, type[BenchmarkTask]] = {
    "mmlu": MMLUTask,
    "hellaswag": HellaSwagTask,
    "generation": GenerationTask,
}
