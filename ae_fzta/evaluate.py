"""Evaluation Module for AE-FZTA.

Computes metrics from Table II (excluding LLM BLEU):
    1. Accuracy — binary anomaly classification
    2. F1-Score — harmonic mean of precision and recall
    3. Trust Prediction AUC — area under ROC for trust inference
    4. Communication Overhead — model parameter size in MB
    5. Latency — mean end-to-end inference time in milliseconds
"""

import logging
import time
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ae_fzta import config
from ae_fzta.data.dataset import ConnectionDataset, GraphDataset
from ae_fzta.decision import make_access_decision
from ae_fzta.models.gnn_trust import GNNTrustModel
from ae_fzta.models.transformer_anomaly import TransformerAnomalyDetector
from ae_fzta.models.llm_policy import LLMPolicyGenerator
from ae_fzta.data.policy_generator import generate_synthetic_policies
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

logger = logging.getLogger(__name__)


def _calculate_action_accuracy(references: List[str], hypotheses: List[str]) -> float:
    """Compute fraction of generated policies with the correct ALLOW/DENY starting token."""
    if not references or not hypotheses:
        return 0.0
    correct = 0
    for ref, hyp in zip(references, hypotheses):
        ref_action = ref.split()[0].upper() if ref.split() else ""
        hyp_action = hyp.split()[0].upper() if hyp.split() else ""
        if ref_action == hyp_action:
            correct += 1
    return float(correct / len(references))

def _calculate_exact_match_rate(references: List[str], hypotheses: List[str]) -> float:
    """Compute fraction of generated policies that match the reference string exactly."""
    if not references or not hypotheses:
        return 0.0
    correct = 0
    for ref, hyp in zip(references, hypotheses):
        if ref.strip().lower() == hyp.strip().lower():
            correct += 1
    return float(correct / len(references))

def evaluate_all(
    transformer: TransformerAnomalyDetector,
    gnn: GNNTrustModel,
    conn_dataset: ConnectionDataset,
    graph_dataset: GraphDataset,
    anomaly_threshold: float = 0.5,
    trust_threshold: float = 0.5,
    batch_size: int = 128,
    latency_samples: int = 100,
    llm_model: LLMPolicyGenerator = None,
) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Args:
        transformer: Trained TBAE model (on GPU).
        gnn: Trained GNTE model (on GPU).
        conn_dataset: Test connection dataset.
        graph_dataset: Test graph dataset.
        anomaly_threshold: τ_a for decision.
        trust_threshold: τ_t for decision.
        batch_size: Eval batch size.
        latency_samples: Samples for latency measurement.

    Returns:
        Dict with accuracy, f1_score, trust_auc, comm_overhead_mb, latency_ms.
        If LLM is present, also returns policy_bleu, action_accuracy (recommended primary), and exact_match_rate.
    """
    device = config.DEVICE
    transformer.eval()
    gnn.eval()

    # --- Accuracy + F1 ---
    all_preds, all_labels = [], []
    loader = torch.utils.data.DataLoader(conn_dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            scores = transformer(batch_x).cpu().tolist()
            all_preds.extend(scores)
            all_labels.extend(batch_y.tolist())

    binary_preds = [1.0 if p >= anomaly_threshold else 0.0 for p in all_preds]
    accuracy = float(accuracy_score(all_labels, binary_preds))
    f1 = float(f1_score(all_labels, binary_preds, average="binary", zero_division=0.0))

    logger.info("Accuracy: %.4f, F1: %.4f", accuracy, f1)

    # --- Trust AUC ---
    graph_data = graph_dataset[0]
    with torch.no_grad():
        trust_scores = gnn(graph_data.x.to(device), graph_data.edge_index.to(device))
        trust_preds = trust_scores.cpu().numpy()
        trust_labels = graph_data.y.numpy()

    try:
        trust_auc = float(roc_auc_score(trust_labels, trust_preds))
    except ValueError:
        trust_auc = 0.5
    logger.info("Trust AUC: %.4f", trust_auc)

    # --- BLEU Score ---
    policy_bleu = None
    if llm_model is not None:
        logger.info("Computing BLEU score for LLM generated policies...")
        syn_logs, syn_refs = generate_synthetic_policies(100, config.POLICY_ALLOW_RATIO, config.RANDOM_SEED)
        generated_pols = [llm_model.generate_policy(log) for log in syn_logs]
        
        references = [[ref.split()] for ref in syn_refs]
        hypotheses = [gen.split() for gen in generated_pols]
        
        smooth_fn = SmoothingFunction().method1
        policy_bleu = float(corpus_bleu(references, hypotheses, smoothing_function=smooth_fn))
        action_acc = _calculate_action_accuracy(syn_refs, generated_pols)
        exact_match = _calculate_exact_match_rate(syn_refs, generated_pols)
        
        logger.info("Policy BLEU: %.4f | Action Acc: %.4f | Exact Match: %.4f", policy_bleu, action_acc, exact_match)
    else:
        action_acc = None
        exact_match = None

    # --- Communication Overhead ---
    total_bytes = 0
    for p in transformer.parameters():
        total_bytes += p.nelement() * p.element_size()
    for p in gnn.parameters():
        total_bytes += p.nelement() * p.element_size()
    if llm_model is not None:
        for p in llm_model.model.parameters():
            total_bytes += p.nelement() * p.element_size()
    overhead_mb = total_bytes / (1024 ** 2)
    logger.info("Comm Overhead: %.4f MB", overhead_mb)

    # --- Latency ---
    num_test = min(latency_samples, len(conn_dataset))
    dummy_logs, _ = generate_synthetic_policies(num_test, config.POLICY_ALLOW_RATIO, config.RANDOM_SEED)
    times = []
    
    latency_trust_scores = []
    
    # Check if we should actually run the loop, if num_test > 0
    if num_test > 0:
        for i in range(num_test):
            feat, _ = conn_dataset[i]
            node_idx = i % graph_data.num_nodes
            start = time.perf_counter()

            with torch.no_grad():
                a_score = transformer(feat.unsqueeze(0).to(device)).item()
                # Feed the graph block, but extract ONLY the decision mapped to node_idx
                t_score = gnn(graph_data.x.to(device), graph_data.edge_index.to(device))[node_idx].item()

            policy_str = "ALLOW"
            if llm_model is not None:
                policy_str = llm_model.generate_policy(dummy_logs[i])

            make_access_decision(a_score, t_score, policy_str, anomaly_threshold, trust_threshold)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            latency_trust_scores.append(t_score)
            
        min_trust = min(latency_trust_scores)
        max_trust = max(latency_trust_scores)
        logger.debug("Latency trust scores sweep: min=%.4f, max=%.4f", min_trust, max_trust)
        if num_test > 1 and min_trust == max_trust:
            logger.warning(
                "Trust scores are constant (%.4f) across all %d latency samples. "
                "This can occur with untrained or early-round models and does not "
                "block evaluation — latency measurement still proceeds.",
                min_trust, num_test,
            )

    latency = float(np.mean(times))
    logger.info("Latency: %.2f ms", latency)

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "trust_auc": trust_auc,
        "comm_overhead_mb": overhead_mb,
        "latency_ms": latency,
        "policy_bleu": policy_bleu,
        "action_accuracy": action_acc,
        "exact_match_rate": exact_match,
    }


def print_results_table(metrics: Dict[str, float]) -> None:
    """Print metrics as a formatted table."""
    units = {
        "accuracy": "ratio", "f1_score": "ratio", "trust_auc": "ratio",
        "comm_overhead_mb": "MB", "latency_ms": "ms",
    }
    header = f"{'Metric':<25} {'Value':>12} {'Unit':>8}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    
    # Print numerical standard metrics
    for k in ["accuracy", "f1_score", "trust_auc", "comm_overhead_mb", "latency_ms"]:
        if k in metrics:
            lines.append(f"{k:<30} {metrics[k]:>10.4f} {units.get(k, ''):>7}")
            
    # Print LLM specific metrics
    act_acc = f"{metrics['action_accuracy']:>10.4f}" if metrics.get("action_accuracy") is not None else "       N/A"
    lines.append(f"{'action_accuracy':<30} {act_acc} {'':>7}")
    
    exact_match = f"{metrics['exact_match_rate']:>10.4f}" if metrics.get("exact_match_rate") is not None else "       N/A"
    lines.append(f"{'exact_match_rate':<30} {exact_match} {'':>7}")
    
    bleu_val = f"{metrics['policy_bleu']:>10.4f}" if metrics.get("policy_bleu") is not None else "       N/A"
    lines.append(f"{'policy_bleu (comparable)':<30} {bleu_val} {'':>7}")
    
    lines.append(sep)
    table = "\n".join(lines)
    logger.info("Results:\n%s", table)
    print("\n" + table)
