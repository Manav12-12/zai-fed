"""Integration Test for AE-FZTA.

End-to-end verification using synthetic data (to keep runtime fast).
Tests: imports, config, 2-round FL training, evaluation, decision.
"""

# ── Make this script work when run directly as `python ae_fzta/integration_test.py`
# Without this, Python can't find the ae_fzta package unless you've done
# `pip install -e .` or set PYTHONPATH.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..")))
# ──────────────────────────────────────────────────────────────────────────────

import logging
import math
import sys
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the full integration test."""

    # --- Imports ---
    from ae_fzta.config import (
        ANOMALY_THRESHOLD, DEVICE, GNN_HIDDEN_DIM, GNN_NUM_LAYERS,
        GNN_DROPOUT, TRANSFORMER_D_MODEL, TRANSFORMER_DIM_FEEDFORWARD,
        TRANSFORMER_DROPOUT, TRANSFORMER_NHEAD, TRANSFORMER_NUM_LAYERS,
        TRUST_THRESHOLD, validate_config,
    )
    from ae_fzta.data.preprocessor import (
        generate_synthetic_data, split_non_iid,
    )
    from ae_fzta.data.dataset import ConnectionDataset, GraphDataset
    from ae_fzta.models.transformer_anomaly import TransformerAnomalyDetector
    from ae_fzta.models.gnn_trust import GNNTrustModel
    from ae_fzta.federated.client import ZTAFederatedClient, encrypt_weights, decrypt_weights
    from ae_fzta.federated.server import fedavg_aggregate, FederatedCoordinator
    from ae_fzta.decision import make_access_decision, make_batch_decisions
    from ae_fzta.train import run_federated_training
    from ae_fzta.evaluate import evaluate_all, print_results_table

    logger.info("All imports successful.")
    validate_config()
    logger.info("Config validation passed (GPU: %s).", DEVICE)

    # --- Training (synthetic, 2 rounds) ---
    logger.info("Starting 2-round federated training on synthetic data...")
    train_results = run_federated_training(
        use_synthetic=True,
        synth_samples=500,
        num_clients=2,
        num_rounds=2,
        local_epochs=1,
        batch_size=64,
        checkpoint_every=2,
    )

    assert train_results["rounds_completed"] == 2
    assert len(train_results["per_round_losses"]) == 2
    logger.info("Training complete.")

    # --- Evaluation ---
    logger.info("Running evaluation...")
    import torch
    num_features = 122  # Match synthetic

    (X, y), (edge_idx, node_feat, trust_lbl) = generate_synthetic_data(
        num_samples=200, num_features=num_features, seed=99,
    )
    cd = ConnectionDataset(X, y)
    gd = GraphDataset(edge_idx, node_feat, trust_lbl)

    transformer = TransformerAnomalyDetector(
        input_dim=num_features,
        d_model=TRANSFORMER_D_MODEL, nhead=TRANSFORMER_NHEAD,
        num_layers=TRANSFORMER_NUM_LAYERS,
        dim_feedforward=TRANSFORMER_DIM_FEEDFORWARD,
        dropout=TRANSFORMER_DROPOUT,
    ).to(DEVICE)

    gnn = GNNTrustModel(
        input_dim=node_feat.shape[1],
        hidden_dim=GNN_HIDDEN_DIM, num_layers=GNN_NUM_LAYERS,
        dropout=GNN_DROPOUT,
    ).to(DEVICE)

    metrics = evaluate_all(
        transformer, gnn, cd, gd,
        anomaly_threshold=ANOMALY_THRESHOLD,
        trust_threshold=TRUST_THRESHOLD,
        latency_samples=5,
    )
    print_results_table(metrics)

    for k, v in metrics.items():
        if v is None:
            continue  # LLM metrics are None when no LLM is loaded
        assert isinstance(v, float) and math.isfinite(v), f"Metric {k}: {v}"
    logger.info("All evaluation metrics finite.")

    # --- Decision ---
    d = make_access_decision(0.2, 0.8, "ALLOW", ANOMALY_THRESHOLD, TRUST_THRESHOLD)
    assert isinstance(d, bool)
    logger.info("Decision function OK.")

    # --- Training metrics ---
    for k in ("final_accuracy", "final_trust_auc", "total_comm_overhead_mb"):
        assert math.isfinite(train_results[k])



    print("\n✅ INTEGRATION TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.error("INTEGRATION TEST FAILED: %s", exc)
        traceback.print_exc()
        sys.exit(1)
