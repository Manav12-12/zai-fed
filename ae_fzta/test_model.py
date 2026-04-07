"""Post-Training Model Testing Script for AE-FZTA.

Loads a trained checkpoint and runs full evaluation on a real dataset,
producing evaluation metrics, a classification report, a confusion matrix,
and sample access decisions.

Usage:
    python3 -m ae_fzta.test_model --checkpoint checkpoints/round_10.npz --dataset nsl
    python3 -m ae_fzta.test_model --checkpoint checkpoints/round_10.npz --dataset ton
    python3 -m ae_fzta.test_model --help
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report

from ae_fzta import config
from ae_fzta.data.dataset import ConnectionDataset, GraphDataset
from ae_fzta.data.preprocessor import (
    build_trust_graph,
    load_nsl_kdd,
    load_ton_iot,
)
from ae_fzta.decision import make_access_decision
from ae_fzta.evaluate import evaluate_all, print_results_table
from ae_fzta.models.gnn_trust import GNNTrustModel
from ae_fzta.models.transformer_anomaly import TransformerAnomalyDetector
from ae_fzta.visualize import plot_confusion_matrix

logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str,
    transformer: TransformerAnomalyDetector,
    gnn: GNNTrustModel,
) -> Dict[str, Any]:
    """Load model weights from a checkpoint file.

    The checkpoint `.npz` contains:
        - arr_0, arr_1, ... : model parameter arrays (transformer first, then GNN)
        - round: FL round number
        - acc: accuracy at checkpoint
        - auc: trust AUC at checkpoint

    Args:
        checkpoint_path: Path to the `.npz` checkpoint file.
        transformer: Transformer model instance to load weights into.
        gnn: GNN model instance to load weights into.

    Returns:
        Metadata dict with 'round', 'acc', 'auc' from the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # allow_pickle=False: .npz files saved with np.savez() store only raw numpy arrays
    # and do not require pickle. Disabling pickle prevents arbitrary code execution
    # from a crafted checkpoint file (CRIT-005).
    data = np.load(checkpoint_path, allow_pickle=False)

    # Identify parameter arrays (arr_0, arr_1, ...) vs metadata (round, acc, auc, val_acc, val_auc)
    metadata_keys = {"round", "acc", "auc", "val_acc", "val_auc"}
    param_keys = sorted(
        [k for k in data.files if k not in metadata_keys],
        key=lambda k: int(k.split("_")[1]) if "_" in k else 0,
    )

    param_arrays = [data[k] for k in param_keys]

    # Split into transformer and GNN parameters
    t_param_count = len(list(transformer.parameters()))
    t_params = param_arrays[:t_param_count]
    g_params = param_arrays[t_param_count:]

    transformer.set_numpy_parameters(t_params)
    gnn.set_numpy_parameters(g_params)

    metadata = {
        "round": int(data["round"]) if "round" in data.files else -1,
        "acc": float(data["acc"]) if "acc" in data.files else -1.0,
        "auc": float(data["auc"]) if "auc" in data.files else -1.0,
    }

    logger.info(
        "Checkpoint loaded: %s (round=%d, acc=%.4f, auc=%.4f)",
        checkpoint_path, metadata["round"], metadata["acc"], metadata["auc"],
    )
    return metadata


def run_sample_decisions(
    transformer: TransformerAnomalyDetector,
    gnn: GNNTrustModel,
    conn_dataset: ConnectionDataset,
    graph_dataset: GraphDataset,
    num_samples: int = 10,
    anomaly_threshold: float = 0.5,
    trust_threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """Run access decisions on individual samples and return detailed results.

    For each sample, computes the anomaly score, trust score, and final
    allow/deny decision with reasoning.

    Args:
        transformer: Trained Transformer model (on GPU).
        gnn: Trained GNN model (on GPU).
        conn_dataset: Test connection dataset.
        graph_dataset: Test graph dataset.
        num_samples: Number of samples to evaluate.
        anomaly_threshold: τ_a for decision.
        trust_threshold: τ_t for decision.

    Returns:
        List of dicts with keys: sample_idx, true_label, anomaly_score,
        trust_score, decision, reasoning.
    """
    device = config.DEVICE
    transformer.eval()
    gnn.eval()

    # Get trust scores for the graph
    graph_data = graph_dataset[0]
    with torch.no_grad():
        trust_scores = gnn(
            graph_data.x.to(device),
            graph_data.edge_index.to(device),
        ).cpu().numpy()

    # Average trust score for sample decisions
    avg_trust = float(trust_scores.mean())

    results = []
    num_samples = min(num_samples, len(conn_dataset))
    # Pick fixed samples based on config random seed for deterministic output
    rng = np.random.RandomState(config.RANDOM_SEED)
    sample_indices = rng.choice(len(conn_dataset), size=num_samples, replace=False)

    for idx in sample_indices:
        feat, label = conn_dataset[idx]
        with torch.no_grad():
            anomaly_score = transformer(feat.unsqueeze(0).to(device)).item()

        # Use a representative trust score (node 0 or average)
        trust_score = min(max(float(trust_scores[idx % len(trust_scores)]), 0.0), 1.0)

        policy_str = "ALLOW"
        decision = make_access_decision(
            anomaly_score, trust_score, policy_str,
            anomaly_threshold, trust_threshold,
        )

        # Build reasoning
        reasons = []
        if anomaly_score >= anomaly_threshold:
            reasons.append(f"anomaly={anomaly_score:.4f} ≥ τ_a={anomaly_threshold}")
        else:
            reasons.append(f"anomaly={anomaly_score:.4f} < τ_a={anomaly_threshold} ✓")

        if trust_score <= trust_threshold:
            reasons.append(f"trust={trust_score:.4f} ≤ τ_t={trust_threshold}")
        else:
            reasons.append(f"trust={trust_score:.4f} > τ_t={trust_threshold} ✓")
            
        if "ALLOW" not in policy_str:
            reasons.append(f"policy_denied={policy_str}")

        true_label_str = "Attack" if label.item() > 0.5 else "Normal"
        decision_str = "ALLOW" if decision else "DENY"

        is_correct = (decision and label.item() <= 0.5) or (not decision and label.item() > 0.5)

        results.append({
            "sample_idx": int(idx),
            "true_label": true_label_str,
            "anomaly_score": anomaly_score,
            "trust_score": trust_score,
            "decision": decision_str,
            "reasoning": " | ".join(reasons),
            "is_correct": is_correct,
            "reasons_raw": reasons
        })

    return results


def test_model(
    checkpoint_path: str,
    dataset: str = "nsl",
    anomaly_threshold: float = 0.5,
    trust_threshold: float = 0.5,
    num_sample_decisions: int = 10,
    output_dir: str = "./results/",
    **kwargs
) -> Dict[str, Any]:
    """Full post-training test pipeline.

    Loads a checkpoint, builds models, runs evaluation, generates
    classification report, confusion matrix, and sample decisions.

    Args:
        checkpoint_path: Path to the `.npz` checkpoint file.
        dataset: "nsl" for NSL-KDD, "ton" for TON_IoT.
        anomaly_threshold: τ_a threshold for decisions.
        trust_threshold: τ_t threshold for decisions.
        num_sample_decisions: Number of sample decisions to print.
        output_dir: Directory for result plots.
        show_errors_only: Only show incorrectly decided samples.
        threshold_sweep: Run a sweep of anomaly thresholds.

    Returns:
        Combined results dict with all metrics and decisions.
    """
    import pandas as pd

    # --- Load data ---
    if dataset == "nsl":
        logger.info("Loading NSL-KDD test dataset...")
        X_train, y_train, X_test, y_test = load_nsl_kdd(
            config.NSL_TRAIN_PATH,
            config.NSL_TEST_PATH,
            config.NSL_COLUMN_NAMES,
            config.NSL_CATEGORICAL_COLS,
        )
        # Build trust graph from TON_IoT
        logger.info("Building trust graph from TON_IoT IPs...")
        df_ton = pd.read_csv(config.TON_TRAIN_PATH)
        df_ton.columns = [c.strip("\ufeff").strip() for c in df_ton.columns]
        df_graph = df_ton.sample(n=min(50000, len(df_ton)), random_state=config.RANDOM_SEED)
        edge_idx, node_feat, trust_lbl = build_trust_graph(
            df_graph, "src_ip", "dst_ip", "label",
            feature_dim=config.GNN_INPUT_DIM,
        )
        actual_input_dim = X_test.shape[1]
    elif dataset == "ton":
        logger.info("Loading TON_IoT test dataset...")
        X_all, y_all = load_ton_iot(
            config.TON_TRAIN_PATH,
            config.TON_DROP_COLS,
            config.TON_CATEGORICAL_COLS,
        )
        split_idx = int(len(X_all) * 0.8)
        perm = np.random.RandomState(config.RANDOM_SEED).permutation(len(X_all))
        X_all, y_all = X_all[perm], y_all[perm]
        X_test, y_test = X_all[split_idx:], y_all[split_idx:]
        # Build trust graph
        df_ton = pd.read_csv(config.TON_TRAIN_PATH)
        df_ton.columns = [c.strip("\ufeff").strip() for c in df_ton.columns]
        df_graph = df_ton.sample(n=min(50000, len(df_ton)), random_state=config.RANDOM_SEED)
        edge_idx, node_feat, trust_lbl = build_trust_graph(
            df_graph, "src_ip", "dst_ip", "label",
            feature_dim=config.GNN_INPUT_DIM,
        )
        actual_input_dim = X_test.shape[1]
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'nsl' or 'ton'.")

    actual_gnn_dim = node_feat.shape[1]
    logger.info("Test data: %d samples, %d features", len(X_test), actual_input_dim)

    # --- Create models ---
    transformer = TransformerAnomalyDetector(
        input_dim=actual_input_dim,
        d_model=config.TRANSFORMER_D_MODEL,
        nhead=config.TRANSFORMER_NHEAD,
        num_layers=config.TRANSFORMER_NUM_LAYERS,
        dim_feedforward=config.TRANSFORMER_DIM_FEEDFORWARD,
        dropout=config.TRANSFORMER_DROPOUT,
    ).to(config.DEVICE)

    gnn = GNNTrustModel(
        input_dim=actual_gnn_dim,
        hidden_dim=config.GNN_HIDDEN_DIM,
        num_layers=config.GNN_NUM_LAYERS,
        dropout=config.GNN_DROPOUT,
    ).to(config.DEVICE)

    # --- Load checkpoint ---
    ckpt_meta = load_checkpoint(checkpoint_path, transformer, gnn)

    # --- Load LLM model ---
    llm_model = None
    try:
        from ae_fzta.train_policy import load_trained_policy_model
        llm_model = load_trained_policy_model(config.BEST_LLM_CHECKPOINT_PATH)
        logger.info("Successfully loaded tuned LLM model from %s.", config.BEST_LLM_CHECKPOINT_PATH)
    except Exception as e:
        logger.warning("No best tuned model found or load failed at %s. Proceeding without LLM metrics. Error: %s", config.BEST_LLM_CHECKPOINT_PATH, str(e))
        llm_model = None

    # --- Create datasets ---
    conn_dataset = ConnectionDataset(X_test, y_test)
    graph_dataset = GraphDataset(edge_idx, node_feat, trust_lbl)

    # --- Run evaluation ---
    logger.info("Running evaluation...")
    metrics = evaluate_all(
        transformer, gnn, conn_dataset, graph_dataset,
        anomaly_threshold=anomaly_threshold,
        trust_threshold=trust_threshold,
        batch_size=config.BATCH_SIZE,
        latency_samples=config.LATENCY_SAMPLES,
        llm_model=llm_model,
    )
    print_results_table(metrics)

    # --- Classification report ---
    logger.info("Generating classification report...")
    all_preds = []
    loader = torch.utils.data.DataLoader(conn_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    transformer.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(config.DEVICE)
            scores = transformer(batch_x).cpu().tolist()
            all_preds.extend(scores)

    binary_preds = np.array([1 if p >= anomaly_threshold else 0 for p in all_preds])
    y_true = y_test

    report = classification_report(
        y_true, binary_preds,
        target_names=["Normal", "Attack"],
        digits=4,
    )
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # --- Confusion matrix ---
    logger.info("Generating confusion matrix...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cm_path = plot_confusion_matrix(
        y_true, binary_preds,
        output_path=str(Path(output_dir) / "test_confusion_matrix.png"),
    )
    print(f"Confusion matrix saved: {cm_path}")

    # --- Sample decisions ---
    logger.info("Running sample access decisions...")
    decisions = run_sample_decisions(
        transformer, gnn, conn_dataset, graph_dataset,
        num_samples=num_sample_decisions,
        anomaly_threshold=anomaly_threshold,
        trust_threshold=trust_threshold,
    )

    print("\n" + "=" * 80)
    print("SAMPLE ACCESS DECISIONS")
    print("=" * 80)
    header = f"{'Idx':>6} {'Label':>8} {'Anomaly':>10} {'Trust':>8} {'Decision':>10}  Reasoning"
    print(header)
    print("-" * len(header) + "-" * 20)
    # Optional show_errors_only filtering
    display_decisions = decisions
    if kwargs.get("show_errors_only"):
        display_decisions = [d for d in decisions if not d["is_correct"]]
    
    for d in display_decisions:
        print(
            f"{d['sample_idx']:>6} {d['true_label']:>8} "
            f"{d['anomaly_score']:>10.4f} {d['trust_score']:>8.4f} "
            f"{d['decision']:>10}  {d['reasoning']}"
        )

    # --- Summary ---
    total_shown = len(display_decisions)
    granted = sum(1 for d in display_decisions if d['decision'] == 'ALLOW')
    denied = sum(1 for d in display_decisions if d['decision'] == 'DENY')
    correct_shown = sum(1 for d in display_decisions if d['is_correct'])
    
    reasons_anomaly = 0
    reasons_trust = 0
    reasons_policy = 0
    reasons_multiple = 0
    
    for d in display_decisions:
        if d['decision'] == 'DENY':
            failures = sum(1 for r in d['reasons_raw'] if "≥" in r or "≤" in r or "policy_denied" in r)
            if failures > 1:
                reasons_multiple += 1
            elif any("≥" in r for r in d['reasons_raw']):
                reasons_anomaly += 1
            elif any("≤" in r for r in d['reasons_raw']):
                reasons_trust += 1
            elif any("policy_denied" in r for r in d['reasons_raw']):
                reasons_policy += 1

    percent_correct = (correct_shown / total_shown * 100) if total_shown > 0 else 0.0
    
    print("\n" + "-" * 80)
    print(f"SUMMARY STATISTICS ({'Filtered to Errors' if kwargs.get('show_errors_only') else 'All Shown Samples'})")
    print(f"Total Cases Shown:           {total_shown}")
    print(f"Correctly Decided:           {correct_shown} ({percent_correct:.1f}%)")
    print(f"Granted Access (ALLOW):      {granted}")
    print(f"Denied Access (DENY):        {denied}")
    print("\nDenial Reasons Breakdown:")
    print(f" - Anomaly Too High:         {reasons_anomaly}")
    print(f" - Trust Too Low:            {reasons_trust}")
    if llm_model is not None:
        print(f" - Policy Denied:            {reasons_policy}")
    print(f" - Multiple Failures:        {reasons_multiple}")
    
    if kwargs.get("threshold_sweep"):
        print("\n" + "=" * 80)
        print("THRESHOLD SWEEP ANALYSIS (Anomaly Threshold τ_a)")
        print("=" * 80)
        sweep_vals = [0.2, 0.3, 0.4, 0.5, 0.6]
        print(f"{'Threshold':<15} {'Accuracy':<15} {'FPR (Blocked Normal)':<25} {'FNR (Missed Attack)':<25}")
        print("-" * 80)
        
        for sweep_t in sweep_vals:
            # We must re-score the pipeline without evaluating the LLM again for speed
            sweep_correct = 0
            sweep_fp = 0
            sweep_fn = 0
            total_n = 0
            total_a = 0
            
            for base_d in decisions:
                a_score = base_d['anomaly_score']
                t_score = base_d['trust_score']
                allow_policy = "ALLOW" not in str(base_d['reasons_raw'])
                
                # Resimulate decision 
                dec = (a_score < sweep_t) and (t_score > trust_threshold) and allow_policy
                is_attack = base_d["true_label"] == "Attack"
                
                if is_attack:
                    total_a += 1
                    if dec: sweep_fn += 1
                    else: sweep_correct += 1
                else:
                    total_n += 1
                    if not dec: sweep_fp += 1
                    else: sweep_correct += 1
                    
            fpr = (sweep_fp / total_n * 100) if total_n > 0 else 0
            fnr = (sweep_fn / total_a * 100) if total_a > 0 else 0
            acc = (sweep_correct / len(decisions)) * 100
            
            pointer = " <-- Current" if abs(sweep_t - anomaly_threshold) < 0.01 else ""
            print(f"{sweep_t:<15.2f} {acc:<15.2f}% {fpr:<25.2f}% {fnr:<25.2f}%{pointer}")

    all_results = {
        **metrics,
        "checkpoint": checkpoint_path,
        "checkpoint_round": ckpt_meta["round"],
        "checkpoint_acc": ckpt_meta["acc"],
        "checkpoint_auc": ckpt_meta["auc"],
        "dataset": dataset,
        "test_samples": len(X_test),
        "sample_decisions": decisions,
    }

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Test a trained AE-FZTA model from a checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m ae_fzta.test_model --checkpoint checkpoints/round_10.npz --dataset nsl
  python3 -m ae_fzta.test_model --checkpoint checkpoints/round_5.npz --dataset ton
  python3 -m ae_fzta.test_model --checkpoint checkpoints/round_10.npz --anomaly-threshold 0.3 --trust-threshold 0.6
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to checkpoint .npz file (e.g. checkpoints/round_10.npz)",
    )
    parser.add_argument(
        "--dataset", type=str, default="nsl", choices=["nsl", "ton"],
        help="Dataset to evaluate on: 'nsl' (NSL-KDD) or 'ton' (TON_IoT). Default: nsl",
    )
    parser.add_argument(
        "--anomaly-threshold", type=float, default=config.ANOMALY_THRESHOLD,
        help=f"Anomaly score threshold τ_a (default: {config.ANOMALY_THRESHOLD})",
    )
    parser.add_argument(
        "--trust-threshold", type=float, default=config.TRUST_THRESHOLD,
        help=f"Trust score threshold τ_t (default: {config.TRUST_THRESHOLD})",
    )
    parser.add_argument(
        "--num-samples", type=int, default=config.TEST_DISPLAY_SAMPLES,
        help=f"Number of sample access decisions to print (default: {config.TEST_DISPLAY_SAMPLES})",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/",
        help="Directory for result plots (default: ./results/)",
    )
    parser.add_argument(
        "--show-errors-only", action="store_true",
        help="Only display incorrectly predicted sample outputs",
    )
    parser.add_argument(
        "--threshold-sweep", action="store_true",
        help="Run an anomaly threshold sweep and output the FPR/FNR comparison table.",
    )

    args = parser.parse_args()

    if args.show_errors_only or args.threshold_sweep:
        pass # Note: this args are injected into the internal function via kwargs replacement, but I already patched it.
        
    results = test_model(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        anomaly_threshold=args.anomaly_threshold,
        trust_threshold=args.trust_threshold,
        num_sample_decisions=args.num_samples,
        output_dir=args.output_dir,
        show_errors_only=args.show_errors_only,
        threshold_sweep=args.threshold_sweep
    )

    logger.info("✅ test_model.py complete.")
