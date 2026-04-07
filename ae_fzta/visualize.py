"""Training Visualization Module for AE-FZTA.

Generates publication-quality plots for federated training results:
    1. Training curves — loss, accuracy, trust AUC per round
    2. Client data distribution — non-IID split visualisation
    3. Confusion matrix — anomaly detection performance
    4. Communication overhead — cumulative per round

All plots are saved to a configurable output directory (default: ./results/).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy-import matplotlib to avoid import overhead when not plotting
_MPL_IMPORTED = False


def _ensure_mpl():
    """Lazy-import matplotlib with Agg backend for headless rendering."""
    global _MPL_IMPORTED
    if not _MPL_IMPORTED:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for server/GPU boxes
        _MPL_IMPORTED = True


def plot_training_curves(
    per_round_losses: List[float],
    per_round_accuracy: List[float],
    per_round_trust_auc: List[float],
    output_path: str = "./results/training_curves.png",
) -> str:
    """Plot loss, accuracy, and trust AUC curves across FL rounds.

    Creates a 1×3 subplot figure showing the progression of each metric
    over federated training rounds.

    Args:
        per_round_losses: Average training loss per round.
        per_round_accuracy: Anomaly detection accuracy per round.
        per_round_trust_auc: GNN trust AUC per round.
        output_path: Path to save the figure.

    Returns:
        Absolute path to the saved figure.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    num_rounds = len(per_round_losses)
    rounds = list(range(1, num_rounds + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("AE-FZTA Federated Training Curves", fontsize=16, fontweight="bold")

    # Loss curve
    axes[0].plot(rounds, per_round_losses, "o-", color="#e74c3c", linewidth=2, markersize=6)
    axes[0].set_title("Training Loss per Round", fontsize=13)
    axes[0].set_xlabel("FL Round")
    axes[0].set_ylabel("Average Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(rounds)

    # Accuracy curve
    axes[1].plot(rounds, per_round_accuracy, "s-", color="#2ecc71", linewidth=2, markersize=6)
    axes[1].set_title("Anomaly Detection Accuracy", fontsize=13)
    axes[1].set_xlabel("FL Round")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(rounds)

    # Trust AUC curve
    axes[2].plot(rounds, per_round_trust_auc, "D-", color="#3498db", linewidth=2, markersize=6)
    axes[2].set_title("Trust Prediction AUC", fontsize=13)
    axes[2].set_xlabel("FL Round")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(rounds)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Training curves saved: %s", output_path)
    return os.path.abspath(output_path)


def plot_client_distribution(
    client_data_sizes: List[int],
    client_attack_ratios: List[float],
    output_path: str = "./results/client_distribution.png",
) -> str:
    """Plot the non-IID data distribution across federated clients.

    Creates a dual-axis bar chart showing sample count (bars) and
    attack ratio (line) per client.

    Args:
        client_data_sizes: Number of training samples per client.
        client_attack_ratios: Fraction of attack samples per client.
        output_path: Path to save the figure.

    Returns:
        Absolute path to the saved figure.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    num_clients = len(client_data_sizes)
    client_labels = [f"Client {i}" for i in range(num_clients)]
    x = np.arange(num_clients)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Non-IID Data Distribution Across Clients", fontsize=15, fontweight="bold")

    # Bar chart for sample counts
    colors = plt.cm.Set2(np.linspace(0, 1, num_clients))
    bars = ax1.bar(x, client_data_sizes, color=colors, alpha=0.8, edgecolor="gray", width=0.6)
    ax1.set_xlabel("Federated Client", fontsize=12)
    ax1.set_ylabel("Number of Samples", fontsize=12, color="#2c3e50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(client_labels)
    ax1.tick_params(axis="y", labelcolor="#2c3e50")

    # Add value labels on bars
    for bar, size in zip(bars, client_data_sizes):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(client_data_sizes) * 0.01,
            f"{size:,}", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # Line chart for attack ratios on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, client_attack_ratios, "ro-", linewidth=2, markersize=10, label="Attack Ratio")
    ax2.set_ylabel("Attack Ratio", fontsize=12, color="#e74c3c")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # Add ratio labels
    for i, ratio in enumerate(client_attack_ratios):
        ax2.annotate(
            f"{ratio:.1%}", (i, ratio), textcoords="offset points",
            xytext=(0, 12), ha="center", fontsize=10, color="#e74c3c",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Client distribution saved: %s", output_path)
    return os.path.abspath(output_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: str = "./results/confusion_matrix.png",
) -> str:
    """Plot a confusion matrix heatmap for anomaly detection.

    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        class_names: Optional labels for classes. Default: ["Normal", "Attack"].
        output_path: Path to save the figure.

    Returns:
        Absolute path to the saved figure.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    if class_names is None:
        class_names = ["Normal", "Attack"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle("Anomaly Detection — Confusion Matrix", fontsize=14, fontweight="bold")

    # Use imshow for heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Ticks and labels
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)

    # Annotate cells with counts and percentages
    total = cm.sum()
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = cm[i, j]
            pct = count / total * 100
            ax.text(
                j, i, f"{count:,}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=12,
                color="white" if count > thresh else "black",
            )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Confusion matrix saved: %s", output_path)
    return os.path.abspath(output_path)


def plot_communication_overhead(
    per_round_overhead_mb: List[float],
    num_clients: int,
    output_path: str = "./results/communication_overhead.png",
) -> str:
    """Plot per-round and cumulative communication overhead.

    Args:
        per_round_overhead_mb: Per-round overhead in MB (single direction).
        num_clients: Number of federated clients.
        output_path: Path to save the figure.

    Returns:
        Absolute path to the saved figure.
    """
    _ensure_mpl()
    import matplotlib.pyplot as plt

    num_rounds = len(per_round_overhead_mb)
    rounds = list(range(1, num_rounds + 1))
    cumulative = np.cumsum(per_round_overhead_mb).tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Communication Overhead ({num_clients} Clients)",
        fontsize=15, fontweight="bold",
    )

    # Per-round overhead
    ax1.bar(rounds, per_round_overhead_mb, color="#9b59b6", alpha=0.8, edgecolor="gray")
    ax1.set_title("Per-Round Overhead", fontsize=13)
    ax1.set_xlabel("FL Round")
    ax1.set_ylabel("Overhead (MB)")
    ax1.set_xticks(rounds)
    ax1.grid(True, axis="y", alpha=0.3)

    # Cumulative overhead
    ax2.fill_between(rounds, cumulative, alpha=0.3, color="#3498db")
    ax2.plot(rounds, cumulative, "o-", color="#3498db", linewidth=2, markersize=6)
    ax2.set_title("Cumulative Overhead", fontsize=13)
    ax2.set_xlabel("FL Round")
    ax2.set_ylabel("Total Overhead (MB)")
    ax2.set_xticks(rounds)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Communication overhead saved: %s", output_path)
    return os.path.abspath(output_path)


def save_all_plots(
    results: Dict[str, Any],
    output_dir: str = "./results/",
) -> List[str]:
    """Generate and save all training visualisation plots.

    Expects the results dict from ``train.run_federated_training()``.
    Required keys:
        - per_round_losses: List[float]
        - per_round_accuracy: List[float]
        - per_round_trust_auc: List[float]
        - per_round_overhead_mb: List[float]
        - client_data_sizes: List[int]
        - client_attack_ratios: List[float]
        - num_clients: int (or FL_NUM_CLIENTS from config)

    Args:
        results: Metrics dictionary from training.
        output_dir: Directory to save all plots.

    Returns:
        List of absolute paths to saved plot files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    # Training curves
    if all(k in results for k in ("per_round_losses", "per_round_accuracy", "per_round_trust_auc")):
        p = plot_training_curves(
            results["per_round_losses"],
            results["per_round_accuracy"],
            results["per_round_trust_auc"],
            output_path=os.path.join(output_dir, "training_curves.png"),
        )
        saved_paths.append(p)

    # Client distribution
    if all(k in results for k in ("client_data_sizes", "client_attack_ratios")):
        p = plot_client_distribution(
            results["client_data_sizes"],
            results["client_attack_ratios"],
            output_path=os.path.join(output_dir, "client_distribution.png"),
        )
        saved_paths.append(p)

    # Communication overhead
    if "per_round_overhead_mb" in results:
        num_clients = results.get("num_clients", 3)
        p = plot_communication_overhead(
            results["per_round_overhead_mb"],
            num_clients,
            output_path=os.path.join(output_dir, "communication_overhead.png"),
        )
        saved_paths.append(p)

    logger.info("All plots saved to %s: %d files", output_dir, len(saved_paths))
    return saved_paths


# ──────────────────────────────────────────────────────────────────────
# Module self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Generate plots from synthetic training results
    logger.info("Generating plots from synthetic training data...")

    synthetic_results = {
        "per_round_losses": [0.35, 0.28, 0.22, 0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07],
        "per_round_accuracy": [0.72, 0.78, 0.83, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95, 0.95],
        "per_round_trust_auc": [0.65, 0.70, 0.75, 0.79, 0.82, 0.84, 0.86, 0.87, 0.88, 0.89],
        "per_round_overhead_mb": [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
        "client_data_sizes": [45000, 35000, 25000, 12000, 8000],
        "client_attack_ratios": [0.71, 0.15, 0.55, 0.89, 0.32],
        "num_clients": 5,
    }

    paths = save_all_plots(synthetic_results, output_dir="./results/")

    # Confusion matrix from synthetic predictions
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, 1000)
    # Simulate 90% accuracy
    y_pred = y_true.copy()
    flip_idx = rng.choice(1000, size=100, replace=False)
    y_pred[flip_idx] = 1 - y_pred[flip_idx]

    cm_path = plot_confusion_matrix(y_true, y_pred, output_path="./results/confusion_matrix.png")
    paths.append(cm_path)

    logger.info("✅ visualize.py self-test complete. %d plots saved.", len(paths))
    for p in paths:
        logger.info("  → %s", p)
