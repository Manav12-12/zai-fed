"""Federated Training Pipeline for AE-FZTA.

Loads real datasets (NSL-KDD or TON_IoT), builds trust graphs, splits
data across federated clients, and runs simulated FL rounds with FedAvg.

All models run on GPU.  LLM is excluded.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from ae_fzta import config
from ae_fzta.data.dataset import ConnectionDataset, GraphDataset
from ae_fzta.data.preprocessor import (
    build_trust_graph,
    load_nsl_kdd,
    load_ton_iot,
    split_non_iid,
    generate_synthetic_data,
)
from ae_fzta.federated.client import ZTAFederatedClient
from ae_fzta.federated.server import FederatedCoordinator, fedavg_aggregate
from ae_fzta.visualize import save_all_plots
from ae_fzta.models.gnn_trust import GNNTrustModel
from ae_fzta.models.transformer_anomaly import TransformerAnomalyDetector

logger = logging.getLogger(__name__)


def run_federated_training(
    dataset: str = "nsl",
    num_clients: int = 3,
    num_rounds: int = 10,
    local_epochs: int = 2,
    batch_size: int = 128,
    checkpoint_every: int = 5,
    checkpoint_dir: str = "./checkpoints/",
    use_synthetic: bool = False,
    synth_samples: int = 1000,
) -> Dict[str, Any]:
    """Run the full simulated federated training pipeline.

    Args:
        dataset: "nsl" for NSL-KDD, "ton" for TON_IoT.
        num_clients: Number of FL clients.
        num_rounds: Communication rounds.
        local_epochs: Local epochs per client per round.
        batch_size: Training batch size.
        checkpoint_every: Save checkpoint every N rounds.
        checkpoint_dir: Checkpoint directory.
        use_synthetic: If True, use synthetic data (for testing).
        synth_samples: Number of synthetic samples if use_synthetic.

    Returns:
        Metrics dict: final_accuracy, final_trust_auc,
        total_comm_overhead_mb, rounds_completed, per_round_losses.
    """
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if use_synthetic:
        logger.info("Using synthetic data (%d samples)...", synth_samples)
        (X_train, y_train), (edge_idx, node_feat, trust_lbl) = generate_synthetic_data(
            num_samples=synth_samples,
            num_features=config.TRANSFORMER_INPUT_DIM,
            num_nodes=50,
            num_edges=200,
            seed=config.RANDOM_SEED,
        )
        X_test, y_test = X_train[:200], y_train[:200]  # Use subset for eval
    elif dataset == "nsl":
        logger.info("Loading NSL-KDD dataset...")
        X_train, y_train, X_test, y_test = load_nsl_kdd(
            config.NSL_TRAIN_PATH,
            config.NSL_TEST_PATH,
            config.NSL_COLUMN_NAMES,
            config.NSL_CATEGORICAL_COLS,
        )
        # Build trust graph from TON_IoT (has real IPs)
        logger.info("Building trust graph from TON_IoT IPs...")
        df_ton = pd.read_csv(config.TON_TRAIN_PATH)
        df_ton.columns = [c.strip("\ufeff").strip() for c in df_ton.columns]
        # Subsample for graph efficiency
        df_graph = df_ton.sample(n=min(50000, len(df_ton)), random_state=config.RANDOM_SEED)
        edge_idx, node_feat, trust_lbl = build_trust_graph(
            df_graph, "src_ip", "dst_ip", "label",
            feature_dim=config.GNN_INPUT_DIM,
        )
    elif dataset == "ton":
        logger.info("Loading TON_IoT dataset...")
        X_all, y_all = load_ton_iot(
            config.TON_TRAIN_PATH,
            config.TON_DROP_COLS,
            config.TON_CATEGORICAL_COLS,
        )
        # Split 80/20 for train/test
        split_idx = int(len(X_all) * 0.8)
        perm = np.random.RandomState(config.RANDOM_SEED).permutation(len(X_all))
        X_all, y_all = X_all[perm], y_all[perm]
        X_train, y_train = X_all[:split_idx], y_all[:split_idx]
        X_test, y_test = X_all[split_idx:], y_all[split_idx:]
        # Build trust graph from the same data
        logger.info("Building trust graph from TON_IoT IPs...")
        df_ton = pd.read_csv(config.TON_TRAIN_PATH)
        df_ton.columns = [c.strip("\ufeff").strip() for c in df_ton.columns]
        df_graph = df_ton.sample(n=min(50000, len(df_ton)), random_state=config.RANDOM_SEED)
        edge_idx, node_feat, trust_lbl = build_trust_graph(
            df_graph, "src_ip", "dst_ip", "label",
            feature_dim=config.GNN_INPUT_DIM if dataset == "ton" else config.GNN_INPUT_DIM,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'nsl' or 'ton'.")

    # Carve out 10% of X_test and y_test strictly for validation
    val_split_idx = int(len(X_test) * 0.9)
    X_val, y_val = X_test[val_split_idx:], y_test[val_split_idx:]
    X_test, y_test = X_test[:val_split_idx], y_test[:val_split_idx]

    # Adjust input dim to match actual feature count
    actual_input_dim = X_train.shape[1]
    actual_gnn_dim = node_feat.shape[1]
    logger.info(
        "Data ready: train=%d, test=%d, validation=%d, features=%d, graph_nodes=%d, graph_feat=%d",
        len(X_train), len(X_test), len(X_val), actual_input_dim,
        node_feat.shape[0], actual_gnn_dim,
    )

    # ------------------------------------------------------------------
    # 2. Split across clients
    # ------------------------------------------------------------------
    client_splits = split_non_iid(
        X_train, y_train, num_clients, alpha=config.FL_DIRICHLET_ALPHA,
    )

    # ------------------------------------------------------------------
    # 2.5 Load LLM Policy Model
    # ------------------------------------------------------------------
    llm_model = None
    if getattr(config, "INCLUDE_LLM_IN_FEDERATION", False):
        try:
            from ae_fzta.train_policy import load_trained_policy_model
            llm_model = load_trained_policy_model(config.POLICY_MODEL_SAVE_PATH)
            logger.info("Successfully loaded LLM model for federation.")
        except Exception as e:
            logger.warning("No saved model found or load failed. Setting INCLUDE_LLM_IN_FEDERATION to False. Error: %s", str(e))
            config.INCLUDE_LLM_IN_FEDERATION = False
            llm_model = None

    # ------------------------------------------------------------------
    # 3. Create models and clients
    # ------------------------------------------------------------------
    encryption_key = AESGCM.generate_key(bit_length=256)

    clients: List[ZTAFederatedClient] = []
    for c in range(num_clients):
        c_X, c_y = client_splits[c]

        t_model = TransformerAnomalyDetector(
            input_dim=actual_input_dim,
            d_model=config.TRANSFORMER_D_MODEL,
            nhead=config.TRANSFORMER_NHEAD,
            num_layers=config.TRANSFORMER_NUM_LAYERS,
            dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
            dropout=config.TRANSFORMER_DROPOUT,
        ).to(config.DEVICE)

        g_model = GNNTrustModel(
            input_dim=actual_gnn_dim,
            hidden_dim=config.GNN_HIDDEN_DIM,
            num_layers=config.GNN_NUM_LAYERS,
            dropout=config.GNN_DROPOUT,
        ).to(config.DEVICE)

        cd = ConnectionDataset(c_X, c_y)
        val_cd = ConnectionDataset(X_val, y_val)
        gd = GraphDataset(edge_idx, node_feat, trust_lbl)

        client = ZTAFederatedClient(
            transformer_model=t_model,
            gnn_model=g_model,
            connection_dataset=cd,
            graph_dataset=gd,
            local_epochs=local_epochs,
            learning_rate=config.FL_LEARNING_RATE,
            batch_size=batch_size,
            dp_noise_scale=config.DP_NOISE_SCALE,
            grad_clip_norm=config.FL_GRAD_CLIP_NORM,
            encryption_key=encryption_key,
            fedprox_mu=getattr(config, "FL_FEDPROX_MU", 0.1),
            llm_model=llm_model,
        )
        clients.append(client)
        logger.info("Client %d: %d samples, attack_ratio=%.2f", c, len(c_X), c_y.mean())

    # ------------------------------------------------------------------
    # 4. Federated training loop
    # ------------------------------------------------------------------
    coordinator = FederatedCoordinator(
        num_clients=num_clients,
        num_rounds=num_rounds,
        fraction_fit=config.FL_FRACTION_FIT,
        fraction_evaluate=config.FL_FRACTION_EVALUATE,
    )

    global_weights = clients[0].get_parameters(config={})
    total_comm_overhead_mb = 0.0
    per_round_losses: List[float] = []
    per_round_accuracy: List[float] = []
    per_round_trust_auc: List[float] = []
    per_round_overhead_mb: List[float] = []
    final_accuracy = 0.0
    final_trust_auc = 0.0

    # Track client data distribution for visualisation
    client_data_sizes = [len(client_splits[c][0]) for c in range(num_clients)]
    client_attack_ratios = [float(client_splits[c][1].mean()) for c in range(num_clients)]

    best_accuracy = 0.0
    best_llm_action_accuracy = 0.0

    for rnd in range(1, num_rounds + 1):
        logger.info("=== Round %d/%d ===", rnd, num_rounds)

        # Fit
        client_results = []
        for c_idx, client in enumerate(clients):
            copied_weights = [w.copy() for w in global_weights]
            updated_params, size, metrics = client.fit(copied_weights, config={}, current_round=rnd)
            client_results.append((updated_params, size, metrics))

        # Aggregate
        client_weights = [r[0] for r in client_results]
        client_sizes = [r[1] for r in client_results]
        global_weights = fedavg_aggregate(client_weights, client_sizes)

        t_count = clients[0]._t_param_count
        g_count = clients[0]._g_param_count
        expected_len = t_count + g_count
        if len(global_weights) != expected_len:
            logger.error("Parameter count mismatch: expected %d, got %d", expected_len, len(global_weights))
            raise RuntimeError(f"FedAvg returned corrupted weights! Expected {expected_len} arrays, got {len(global_weights)}.")

        # Overhead
        round_overhead = coordinator.compute_communication_overhead(global_weights)
        round_total_overhead = round_overhead * num_clients
        total_comm_overhead_mb += round_total_overhead
        per_round_overhead_mb.append(round_total_overhead)

        # Average loss
        avg_loss = sum(r[2]["avg_loss"] * r[1] for r in client_results) / sum(client_sizes)
        per_round_losses.append(avg_loss)

        # Evaluate on clients (Training Performance)
        eval_results = []
        for client in clients:
            loss, size, eval_m = client.evaluate(global_weights, config={})
            eval_results.append((size, eval_m))

        avg_metrics = coordinator.weighted_average_metrics(eval_results)
        final_accuracy = avg_metrics.get("accuracy", 0.0)
        final_trust_auc = avg_metrics.get("trust_auc", 0.0)
        per_round_accuracy.append(final_accuracy)
        per_round_trust_auc.append(final_trust_auc)
        
        # Evaluate on strict Validation Subset (Held-Out Performance)
        t_params = global_weights[:t_count]
        g_params = global_weights[t_count:expected_len]

        # Use the first client's models purely as structural shells for the weights
        val_transformer = clients[0].transformer
        val_gnn = clients[0].gnn
        val_transformer.set_numpy_parameters(t_params)
        val_gnn.set_numpy_parameters(g_params)
        
        val_transformer.eval()
        val_gnn.eval()
        
        all_val_preds = []
        all_val_labels = []
        
        val_loader = torch.utils.data.DataLoader(val_cd, batch_size=config.BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(config.DEVICE)
                preds = val_transformer(batch_x).squeeze().cpu().numpy()
                all_val_preds.extend(preds)
                all_val_labels.extend(batch_y.numpy())
                
            val_trust_preds = val_gnn(gd[0].x.to(config.DEVICE), gd[0].edge_index.to(config.DEVICE)).cpu().numpy()

        all_val_preds = np.array(all_val_preds)
        all_val_labels = np.array(all_val_labels)
        val_trust_preds = np.array(val_trust_preds)
        
        # Calculate sklearn metrics explicitly
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Anomaly is anomaly score >= threshold
        val_anomaly_decisions = (all_val_preds >= config.ANOMALY_THRESHOLD).astype(int)
        val_acc = float(accuracy_score(all_val_labels, val_anomaly_decisions))
        
        # AUC needs at least 2 classes in the graph labels, but we'll try/except just in case
        try:
            val_auc = float(roc_auc_score(gd[0].y.numpy(), val_trust_preds))
        except ValueError:
            val_auc = 0.0

        logger.info(
            "Round %d: loss=%.4f, acc=%.4f, val_acc=%.4f, auc=%.4f, val_auc=%.4f, overhead=%.4f MB | LLM fine-tuned locally: %s",
            rnd, avg_loss, final_accuracy, val_acc, final_trust_auc, val_auc, round_overhead, (llm_model is not None)
        )

        # Checkpoint
        if rnd % checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"round_{rnd}.npz")
            np.savez(ckpt_path, *global_weights, round=rnd, acc=final_accuracy, val_acc=val_acc, auc=final_trust_auc, val_auc=val_auc)
            logger.info("Checkpoint saved: %s", ckpt_path)

        # Always save best model against VAL_ACC explicitly!
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_ckpt_path = os.path.join(checkpoint_dir, "best_model.npz")
            np.savez(best_ckpt_path, *global_weights, round=rnd, acc=final_accuracy, val_acc=val_acc, auc=final_trust_auc, val_auc=val_auc)
            logger.info("New best model! val_acc=%.4f (train acc=%.4f). Saved to %s", val_acc, final_accuracy, best_ckpt_path)
            
        # --- Separate LLM Checkpoint ---
        if llm_model is not None:
            # Measure action accuracy on a fresh batch of 20 synthetic pairs
            from ae_fzta.data.policy_generator import generate_synthetic_policies
            syn_logs, syn_refs = generate_synthetic_policies(20, config.POLICY_ALLOW_RATIO, config.RANDOM_SEED + rnd)
            action_correct = 0
            for log, ref in zip(syn_logs, syn_refs):
                generated = llm_model.generate_policy(log)
                gen_action = generated.split()[0].upper() if generated.split() else ""
                ref_action = ref.split()[0].upper() if ref.split() else ""
                if gen_action == ref_action:
                    action_correct += 1
            current_llm_action_accuracy = action_correct / 20.0
            
            # Save if better than previous best
            if current_llm_action_accuracy > best_llm_action_accuracy:
                best_llm_action_accuracy = current_llm_action_accuracy
                
                abs_path = os.path.abspath(config.BEST_LLM_CHECKPOINT_PATH)
                llm_save_dir = Path(abs_path)
                llm_save_dir.mkdir(parents=True, exist_ok=True)
                llm_model.model.save_pretrained(llm_save_dir, safe_serialization=False)
                llm_model.tokenizer.save_pretrained(llm_save_dir, safe_serialization=False)
                logger.info("New best LLM! Action Accuracy: %.2f. Saved to %s", best_llm_action_accuracy, abs_path)

    results = {
        "final_accuracy": final_accuracy,
        "final_val_acc": val_acc,
        "final_trust_auc": final_trust_auc,
        "final_val_auc": val_auc,
        "best_llm_action_accuracy": best_llm_action_accuracy,
        "total_comm_overhead_mb": total_comm_overhead_mb,
        "rounds_completed": num_rounds,
        "per_round_losses": per_round_losses,
        "per_round_accuracy": per_round_accuracy,
        "per_round_trust_auc": per_round_trust_auc,
        "per_round_overhead_mb": per_round_overhead_mb,
        "client_data_sizes": client_data_sizes,
        "client_attack_ratios": client_attack_ratios,
        "num_clients": num_clients,
    }
    logger.info("Training complete: %s", {
        k: v for k, v in results.items()
        if k not in ("per_round_losses", "per_round_accuracy", "per_round_trust_auc",
                     "per_round_overhead_mb", "client_data_sizes", "client_attack_ratios")
    })

    # Generate and save training visualisation plots
    try:
        plot_paths = save_all_plots(results, output_dir="./results/")
        logger.info("Training plots saved: %s", plot_paths)
    except Exception as exc:
        logger.warning("Could not generate plots: %s", exc)

    return results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results = run_federated_training(
        dataset="nsl",
        num_clients=config.FL_NUM_CLIENTS,
        num_rounds=config.FL_NUM_ROUNDS,
        local_epochs=config.FL_LOCAL_EPOCHS,
        batch_size=config.BATCH_SIZE,
        checkpoint_every=config.FL_CHECKPOINT_EVERY,
    )

    logger.info("Final: %s", results)
    logger.info("✅ train.py complete.")
