"""
Task-specific metrics for ConcAdptr benchmarking.

accuracy() and f1_score() have no extra dependencies.
bleu() and rouge() require the evaluate library:
    pip install 'concadptr[benchmarks]'
"""

from __future__ import annotations


def accuracy(preds: list[int], labels: list[int]) -> float:
    """Compute classification accuracy.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.

    Returns:
        Fraction of correct predictions in [0, 1]. Returns 0.0 for empty input.
    """
    if not preds:
        return 0.0
    return sum(p == gt for p, gt in zip(preds, labels)) / len(preds)


def f1_score(preds: list[int], labels: list[int], average: str = "macro") -> float:
    """Compute F1 score from integer class predictions.

    Computes per-class precision and recall from a manual confusion matrix,
    then averages them according to the averaging strategy.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        average: "macro" computes unweighted mean F1 across all classes.
            "binary" computes F1 for the positive class (label == 1) only.

    Returns:
        F1 score in [0, 1]. Returns 0.0 for empty input.
    """
    if not preds:
        return 0.0

    classes = sorted(set(labels) | set(preds))
    if average == "binary":
        classes = [1]

    f1s = []
    for cls in classes:
        tp = sum(p == cls and gt == cls for p, gt in zip(preds, labels))
        fp = sum(p == cls and gt != cls for p, gt in zip(preds, labels))
        fn = sum(p != cls and gt == cls for p, gt in zip(preds, labels))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)

    return sum(f1s) / len(f1s) if f1s else 0.0


def bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus BLEU score using SacreBLEU via the evaluate library.

    Args:
        predictions: Generated text strings.
        references: Reference text strings (one per prediction).

    Returns:
        BLEU score as a percentage in [0, 100].

    Raises:
        ImportError: If the evaluate library is not installed.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError:
        raise ImportError(
            "BLEU metric requires the evaluate library. "
            "Install with: pip install 'concadptr[benchmarks]'"
        )
    metric = hf_evaluate.load("sacrebleu")
    result = metric.compute(predictions=predictions, references=[[r] for r in references])
    return float(result["score"])


def rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE scores using the evaluate library.

    Args:
        predictions: Generated text strings.
        references: Reference text strings (one per prediction).

    Returns:
        Dict with keys "rouge1", "rouge2", "rougeL" — each a float in [0, 1].

    Raises:
        ImportError: If the evaluate library is not installed.
    """
    try:
        import evaluate as hf_evaluate
    except ImportError:
        raise ImportError(
            "ROUGE metric requires the evaluate library. "
            "Install with: pip install 'concadptr[benchmarks]'"
        )
    metric = hf_evaluate.load("rouge")
    result = metric.compute(predictions=predictions, references=references)
    return {
        "rouge1": float(result["rouge1"]),
        "rouge2": float(result["rouge2"]),
        "rougeL": float(result["rougeL"]),
    }
