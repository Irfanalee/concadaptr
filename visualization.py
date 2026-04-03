"""
Visualization utilities for ConcAdptr routing analysis.

Provides tools for understanding how the router distributes tokens
across experts, diagnosing expert collapse, and visualizing
layer-wise routing patterns.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch


def plot_routing_heatmap(
    routing_weights: torch.Tensor,
    expert_names: List[str],
    title: str = "Routing Weights Heatmap",
    save_path: Optional[str] = None,
) -> None:
    """Plot a heatmap of routing weights across tokens and experts.

    Args:
        routing_weights: Tensor of shape (seq_len, num_experts).
        expert_names: Names of the experts for axis labels.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. "
            "Install with: pip install matplotlib"
        )

    weights = routing_weights.cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(expert_names) * 1.5), 6))
    im = ax.imshow(weights.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Expert")
    ax.set_yticks(range(len(expert_names)))
    ax.set_yticklabels(expert_names)
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Routing Weight")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_expert_load(
    routing_stats: Dict[str, torch.Tensor],
    expert_names: List[str],
    title: str = "Expert Load Distribution",
    save_path: Optional[str] = None,
) -> None:
    """Plot the load distribution across experts.

    Args:
        routing_stats: Output from router.get_routing_stats().
        expert_names: Names of the experts.
        title: Plot title.
        save_path: Optional path to save the figure.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        raise ImportError(
            "Visualization requires matplotlib. "
            "Install with: pip install matplotlib"
        )

    expert_load = routing_stats["expert_load"].cpu().numpy()
    utilization = routing_stats.get("expert_utilization", torch.zeros_like(routing_stats["expert_load"])).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Load distribution
    colors = plt.cm.Set2(np.linspace(0, 1, len(expert_names)))
    axes[0].bar(expert_names, expert_load, color=colors)
    axes[0].axhline(y=1.0 / len(expert_names), color="red", linestyle="--", label="Uniform")
    axes[0].set_ylabel("Average Routing Weight")
    axes[0].set_title("Expert Load")
    axes[0].legend()
    axes[0].tick_params(axis="x", rotation=45)

    # Utilization
    axes[1].bar(expert_names, utilization, color=colors)
    axes[1].set_ylabel("Fraction of Tokens (in Top-2)")
    axes[1].set_title("Expert Utilization")
    axes[1].tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_routing_summary(
    routing_stats: Dict[str, torch.Tensor],
    expert_names: List[str],
) -> str:
    """Generate a text summary of routing statistics.

    Args:
        routing_stats: Output from router.get_routing_stats().
        expert_names: Names of the experts.

    Returns:
        Formatted summary string.
    """
    lines = ["ConcAdptr Routing Summary", "=" * 40]

    if "routing_entropy" in routing_stats:
        entropy = routing_stats["routing_entropy"].item()
        max_entropy = torch.log(torch.tensor(float(len(expert_names)))).item()
        lines.append(f"Routing Entropy: {entropy:.4f} / {max_entropy:.4f} (max)")
        lines.append(f"Uniformity: {entropy / max_entropy * 100:.1f}%")

    if "expert_load" in routing_stats:
        lines.append("\nExpert Load:")
        load = routing_stats["expert_load"]
        for name, l in zip(expert_names, load):
            bar = "█" * int(l * 50)
            lines.append(f"  {name:20s} {l:.4f} {bar}")

    if "expert_utilization" in routing_stats:
        lines.append("\nExpert Utilization (top-2):")
        util = routing_stats["expert_utilization"]
        for name, u in zip(expert_names, util):
            bar = "█" * int(u * 50)
            lines.append(f"  {name:20s} {u:.4f} {bar}")

    result = "\n".join(lines)
    print(result)
    return result
