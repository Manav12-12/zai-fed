"""Federated Learning Server for AE-FZTA.

Implements server-side federated aggregation (FedAvg per Paper Eq. 16)
and coordination of the federated training process across edge/cloud nodes.

PROTOTYPE NOTE (CRIT-003 — Gradient Compression): The paper (Section V-F)
reports that "Federated Learning with Gradient Compression" reduces
communication overhead by ~70% (from ~80 MB/round to ~25 MB/round) at only
0.6% accuracy cost.  This prototype implements standard FedAvg *without*
gradient compression.  The current overhead of ~74 MB/round (Transformer
~8.4 MB + GNN ~4 MB × 5 clients × upload+download) corresponds to the
standard FedAvg baseline described in the paper, not the compressed variant.

To implement gradient compression and reproduce the paper's 70% savings:
  - Top-K sparsification: transmit only the largest K gradient elements
    (K ≈ 30% of parameters achieves ~70% byte reduction).
  - Quantisation: reduce float32 → int8 (4× size reduction).
  Both approaches require a matching decompression step in fedavg_aggregate().

Functions:
    fedavg_aggregate: Weighted averaging of client model parameters.

Classes:
    FederatedCoordinator: Server-side FL coordination and strategy building.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.server.strategy import FedAvg

logger = logging.getLogger(__name__)


def fedavg_aggregate(
    client_weights: List[List[np.ndarray]],
    client_sizes: List[int],
) -> List[np.ndarray]:
    """Aggregate client weights using Federated Averaging (Paper Eq. 16).

    w_{t+1} = Σ_k (|D_k| / Σ_j |D_j|) * (w_t + Δw_k)

    Each client's contribution is weighted by its local dataset size
    divided by the total number of samples across all clients.

    Args:
        client_weights: List of weight lists, one per client. Each inner
            list contains numpy arrays (one per model parameter).
        client_sizes: List of dataset sizes, one per client.

    Returns:
        Aggregated weight list with same structure as each client's weights.

    Raises:
        ValueError: If inputs are empty or lengths mismatch.
    """
    if len(client_weights) == 0:
        raise ValueError("client_weights must be non-empty.")
    if len(client_weights) != len(client_sizes):
        raise ValueError(
            f"client_weights ({len(client_weights)}) and client_sizes "
            f"({len(client_sizes)}) must have equal length."
        )
    if any(s <= 0 for s in client_sizes):
        raise ValueError("All client sizes must be positive.")

    total_samples = sum(client_sizes)
    num_params = len(client_weights[0])

    aggregated: List[np.ndarray] = []
    for p_idx in range(num_params):
        weighted_sum = np.zeros_like(client_weights[0][p_idx], dtype=np.float64)
        for c_idx in range(len(client_weights)):
            weight_fraction = client_sizes[c_idx] / total_samples
            weighted_sum += weight_fraction * client_weights[c_idx][p_idx].astype(
                np.float64
            )
        aggregated.append(weighted_sum.astype(client_weights[0][p_idx].dtype))

    logger.info(
        "FedAvg aggregation: %d clients, %d params, total_samples=%d",
        len(client_weights),
        num_params,
        total_samples,
    )
    return aggregated


class FederatedCoordinator:
    """Server-side coordinator for AE-FZTA federated learning.

    Manages strategy construction, metrics aggregation, and communication
    overhead tracking.
    """

    def __init__(
        self,
        num_clients: int,
        num_rounds: int,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
    ) -> None:
        """Initialise the coordinator.

        Args:
            num_clients: Total number of participating clients.
            num_rounds: Number of communication rounds.
            fraction_fit: Fraction of clients to sample for training.
            fraction_evaluate: Fraction of clients to sample for evaluation.
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        logger.info(
            "FederatedCoordinator: clients=%d, rounds=%d, fit_frac=%.2f, eval_frac=%.2f",
            num_clients,
            num_rounds,
            fraction_fit,
            fraction_evaluate,
        )

    def build_strategy(self) -> FedAvg:
        """Build a Flower FedAvg strategy with the configured parameters.

        Minimum client counts equal total client count so all clients
        participate every round.

        Returns:
            A configured flwr.server.strategy.FedAvg instance.
        """
        strategy = FedAvg(
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.num_clients,
            min_evaluate_clients=self.num_clients,
            min_available_clients=self.num_clients,
            evaluate_metrics_aggregation_fn=self._aggregate_metrics_fn,
        )
        logger.info("FedAvg strategy built.")
        return strategy

    def _aggregate_metrics_fn(
        self, metrics: List[Tuple[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Internal wrapper for weighted metric aggregation.

        Args:
            metrics: List of (num_samples, metrics_dict) tuples from clients.

        Returns:
            Dictionary of weighted-average metric values.
        """
        return self.weighted_average_metrics(metrics)

    def compute_communication_overhead(self, weights: List[np.ndarray]) -> float:
        """Compute total byte size of a weight list in megabytes.

        Uses the .nbytes attribute of numpy arrays.

        Args:
            weights: List of numpy arrays representing model parameters.

        Returns:
            Total size in megabytes (MB).
        """
        total_bytes = sum(w.nbytes for w in weights)
        size_mb = total_bytes / (1024 * 1024)
        logger.info(
            "Communication overhead: %d params, %.4f MB",
            len(weights),
            size_mb,
        )
        return size_mb

    @staticmethod
    def weighted_average_metrics(
        metrics_list: List[Tuple[int, Dict[str, float]]]
    ) -> Dict[str, float]:
        """Compute weighted average of metrics across clients.

        Each client's metrics are weighted by its sample count.

        Args:
            metrics_list: List of (num_samples, metrics_dict) tuples.

        Returns:
            Dictionary of weighted-average metric values.
        """
        if not metrics_list:
            return {}

        total_samples = sum(n for n, _ in metrics_list)
        if total_samples == 0:
            return {}

        # Collect all metric keys
        all_keys = set()
        for _, m in metrics_list:
            all_keys.update(m.keys())

        result: Dict[str, float] = {}
        for key in all_keys:
            weighted_sum = 0.0
            for num_samples, m in metrics_list:
                weighted_sum += num_samples * m.get(key, 0.0)
            result[key] = weighted_sum / total_samples

        return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Test fedavg_aggregate with synthetic weights ---
    # Simulate 3 clients with different sizes
    rng = np.random.RandomState(42)
    param_shapes = [(64, 16), (64,), (32, 64), (32,), (1, 32), (1,)]

    client_weights_list: List[List[np.ndarray]] = []
    client_sizes = [100, 50, 150]

    for _ in range(3):
        client_w = [rng.randn(*s).astype(np.float32) for s in param_shapes]
        client_weights_list.append(client_w)

    aggregated = fedavg_aggregate(client_weights_list, client_sizes)

    # Assert output shapes match input shapes
    assert len(aggregated) == len(param_shapes), (
        f"Expected {len(param_shapes)} params, got {len(aggregated)}"
    )
    for agg, shape in zip(aggregated, param_shapes):
        assert agg.shape == shape, f"Expected shape {shape}, got {agg.shape}"

    # Verify weighted average is correct for first param
    expected = (
        100 / 300 * client_weights_list[0][0].astype(np.float64)
        + 50 / 300 * client_weights_list[1][0].astype(np.float64)
        + 150 / 300 * client_weights_list[2][0].astype(np.float64)
    )
    assert np.allclose(aggregated[0], expected.astype(np.float32), atol=1e-5), (
        "Weighted average does not match"
    )

    # --- Test FederatedCoordinator ---
    coordinator = FederatedCoordinator(
        num_clients=3, num_rounds=5, fraction_fit=1.0, fraction_evaluate=1.0
    )

    # Test strategy building
    strategy = coordinator.build_strategy()
    assert strategy is not None

    # Test communication overhead
    overhead = coordinator.compute_communication_overhead(aggregated)
    assert overhead > 0.0, f"Expected positive overhead, got {overhead}"

    # Test weighted average metrics
    metrics_list = [
        (100, {"accuracy": 0.9, "trust_auc": 0.85}),
        (50, {"accuracy": 0.8, "trust_auc": 0.90}),
        (150, {"accuracy": 0.95, "trust_auc": 0.88}),
    ]
    avg_metrics = coordinator.weighted_average_metrics(metrics_list)
    assert "accuracy" in avg_metrics
    assert "trust_auc" in avg_metrics
    # Manual check: (100*0.9 + 50*0.8 + 150*0.95) / 300 = (90+40+142.5)/300 = 0.9083...
    assert abs(avg_metrics["accuracy"] - 0.9083333) < 1e-4, (
        f"Expected ~0.9083, got {avg_metrics['accuracy']}"
    )

    logger.info("Aggregated shapes: %s", [a.shape for a in aggregated])
    logger.info("Overhead: %.4f MB", overhead)
    logger.info("Avg metrics: %s", avg_metrics)
    logger.info("✅ STEP 9 COMPLETE — server.py all tests passed.")
