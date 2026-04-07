"""Data Ingestion and Preprocessing Module (DIPM).

Loads, cleans, and transforms real-world network intrusion datasets
(NSL-KDD and TON_IoT) into tensors suitable for the Transformer and
GNN models.  Also builds trust graphs from connection metadata.

Functions:
    load_nsl_kdd: Load and one-hot encode NSL-KDD CSVs.
    load_ton_iot: Load and clean TON_IoT CSV.
    build_trust_graph: Build edge index + node features from connections.
    split_non_iid: Dirichlet-based non-IID client split.
    split_iid: Simple IID client split.
    generate_synthetic_data: Synthetic fallback (preserved).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# NSL-KDD Loader
# ──────────────────────────────────────────────────────────────────────

def load_nsl_kdd(
    train_path: str,
    test_path: str,
    column_names: List[str],
    categorical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NSL-KDD train/test CSVs and one-hot encode categoricals.

    The .txt files have **no header row**.  Column names come from the
    ARFF attribute list and are passed via ``column_names``.

    Processing steps:
        1. Read headerless CSVs and assign column names.
        2. Drop the ``difficulty`` column (NSL-KDD artefact, not a feature).
        3. Binary label: ``normal`` → 0, everything else → 1.
        4. One-hot encode categorical columns (protocol_type, service, flag).
           Train and test are concatenated before encoding to ensure
           consistent dummy columns, then split back.
        5. Cast to float32.

    Args:
        train_path: Path to KDDTrain+.txt.
        test_path: Path to KDDTest+.txt.
        column_names: Ordered column names including 'label' and 'difficulty'.
        categorical_cols: Column names to one-hot encode.

    Returns:
        (X_train, y_train, X_test, y_test) all as numpy arrays.
        X arrays are float32, y arrays are int64.

    Raises:
        FileNotFoundError: If either file does not exist.
    """
    df_train = pd.read_csv(train_path, header=None, names=column_names)
    df_test = pd.read_csv(test_path, header=None, names=column_names)

    # Combine KDDTrain+ and KDDTest+ to ensure consistent one-hot encoding columns
    n_train = len(df_train)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    logger.info("NSL-KDD loaded: train=%d rows, test=%d rows", n_train, len(df_test))

    # Drop difficulty score — not a network feature
    df_all = df_all.drop(columns=["difficulty"])

    # Binary labels
    df_all["label"] = (df_all["label"] != "normal").astype(int)

    # One-hot encode categorical columns
    df_all = pd.get_dummies(df_all, columns=categorical_cols, dtype=float)
    
    # Split back into precise train and test sets to preserve KDDTest+ zero-day attacks
    df_train_enc = df_all.iloc[:n_train]
    df_test_enc = df_all.iloc[n_train:]

    y_train = df_train_enc.pop("label").values.astype(np.int64)
    y_test = df_test_enc.pop("label").values.astype(np.int64)

    X_train = df_train_enc.values.astype(np.float32)
    X_test = df_test_enc.values.astype(np.float32)

    # Normalise features to zero mean, unit variance.
    # Fit on training data ONLY to prevent data leakage.
    # This is critical: raw NSL-KDD features like src_bytes can reach
    # ~1.4 billion, which causes NaN in the Transformer's linear layers.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # Replace any NaN/inf introduced by zero-variance columns
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "NSL-KDD encoded+normalised: %d features, train_attack_ratio=%.2f, "
        "test_attack_ratio=%.2f, feature_range=[%.2f, %.2f]",
        X_train.shape[1],
        y_train.mean(),
        y_test.mean(),
        X_train.min(),
        X_train.max(),
    )

    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────────────────────────────
# TON_IoT Loader
# ──────────────────────────────────────────────────────────────────────

def load_ton_iot(
    filepath: str,
    drop_cols: List[str],
    categorical_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load TON_IoT network CSV and clean it for training.

    Processing steps:
        1. Read CSV with header row.
        2. Drop string/IP/non-numeric columns.
        3. Replace '-' sentinel values with 0.
        4. One-hot encode remaining categorical columns.
        5. Extract binary label column.
        6. Replace any remaining NaN with 0.
        7. Cast to float32.

    Args:
        filepath: Path to train_test_network.csv.
        drop_cols: Column names to drop entirely.
        categorical_cols: Column names to one-hot encode.

    Returns:
        (X, y) — feature matrix (float32) and label vector (int64).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    df = pd.read_csv(filepath)

    # Remove BOM from first column name if present
    df.columns = [c.strip("\ufeff").strip() for c in df.columns]

    logger.info("TON_IoT loaded: %d rows, %d columns", len(df), len(df.columns))

    # Drop non-numeric / identifier columns
    existing_drop = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    # Replace '-' with 0 in all remaining columns
    df = df.replace("-", 0)

    # Extract label
    y = df.pop("label").values.astype(np.int64)

    # One-hot encode categoricals
    existing_cat = [c for c in categorical_cols if c in df.columns]
    if existing_cat:
        df = pd.get_dummies(df, columns=existing_cat, dtype=float)

    # Coerce everything to numeric, replace NaN with 0
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    X = df.values.astype(np.float32)

    # Normalise to zero mean, unit variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    logger.info(
        "TON_IoT encoded+normalised: %d features, attack_ratio=%.2f, "
        "feature_range=[%.2f, %.2f]",
        X.shape[1], y.mean(), X.min(), X.max(),
    )

    return X, y


# ──────────────────────────────────────────────────────────────────────
# Trust Graph Builder
# ──────────────────────────────────────────────────────────────────────

def build_trust_graph(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    label_col: str,
    feature_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a trust graph from connection source/destination pairs.

    Nodes are unique values from ``src_col`` ∪ ``dst_col``.
    Edges connect src → dst for every connection row.

    Node features are engineered from aggregate statistics:
        - Out-degree (normalised)
        - In-degree (normalised)
        - Fraction of outgoing connections that are malicious
        - Fraction of incoming connections that are malicious
        - Remaining dims filled with random features for diversity

    Trust labels:
        - Nodes with malicious_fraction < 0.5 → trusted (1)
        - Nodes with malicious_fraction ≥ 0.5 → untrusted (0)

    Args:
        df: DataFrame with at least src_col, dst_col, and label_col.
        src_col: Column name for source entities.
        dst_col: Column name for destination entities.
        label_col: Binary label column (0=normal, 1=attack).
        feature_dim: Total node feature dimension.

    Returns:
        (edge_index, node_features, trust_labels):
            - edge_index: int64 array of shape (2, num_edges).
            - node_features: float32 array of shape (num_nodes, feature_dim).
            - trust_labels: float32 array of shape (num_nodes,).
    """
    # Build node vocabulary
    all_entities = pd.concat([df[src_col], df[dst_col]]).unique()
    entity_to_idx = {e: i for i, e in enumerate(all_entities)}
    num_nodes = len(entity_to_idx)

    # Build edge index
    src_indices = df[src_col].map(entity_to_idx).values
    dst_indices = df[dst_col].map(entity_to_idx).values
    edge_index = np.stack([src_indices, dst_indices], axis=0).astype(np.int64)

    # Compute node statistics
    labels = df[label_col].values
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    in_degree = np.zeros(num_nodes, dtype=np.float32)
    out_malicious = np.zeros(num_nodes, dtype=np.float32)
    in_malicious = np.zeros(num_nodes, dtype=np.float32)

    for i in range(len(src_indices)):
        s, d, lbl = src_indices[i], dst_indices[i], labels[i]
        out_degree[s] += 1
        in_degree[d] += 1
        if lbl == 1:
            out_malicious[s] += 1
            in_malicious[d] += 1

    # Normalise
    max_out = max(out_degree.max(), 1.0)
    max_in = max(in_degree.max(), 1.0)
    out_degree_norm = out_degree / max_out
    in_degree_norm = in_degree / max_in
    out_mal_frac = np.divide(out_malicious, out_degree, out=np.zeros_like(out_malicious), where=out_degree > 0)
    in_mal_frac = np.divide(in_malicious, in_degree, out=np.zeros_like(in_malicious), where=in_degree > 0)

    # Build node features
    node_features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
    node_features[:, 0] = out_degree_norm
    node_features[:, 1] = in_degree_norm
    node_features[:, 2] = out_mal_frac
    node_features[:, 3] = in_mal_frac
    # Fill remaining dims with small random features for diversity
    if feature_dim > 4:
        rng = np.random.RandomState(42)
        node_features[:, 4:] = rng.randn(num_nodes, feature_dim - 4).astype(np.float32) * 0.01

    # Trust labels: low malicious fraction → trusted
    total_mal_frac = (out_mal_frac + in_mal_frac) / 2.0
    trust_labels = (total_mal_frac < 0.5).astype(np.float32)

    logger.info(
        "Trust graph: %d nodes, %d edges, trusted_ratio=%.2f",
        num_nodes, edge_index.shape[1], trust_labels.mean(),
    )

    return edge_index, node_features, trust_labels


# ──────────────────────────────────────────────────────────────────────
# Data Splitting for Federated Learning
# ──────────────────────────────────────────────────────────────────────

def split_non_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split data across clients using Dirichlet distribution (non-IID).

    The Dirichlet distribution controls class balance heterogeneity:
        α → 0:   extreme non-IID (each client gets mostly one class)
        α → ∞:   approaches IID (uniform class balance)
        α = 0.5: moderate heterogeneity (default)

    Args:
        X: Feature matrix of shape (N, D).
        y: Label vector of shape (N,).
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter.
        seed: Random seed.

    Returns:
        List of (X_client, y_client) tuples, one per client.
    """
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        # Scale proportions to actual counts
        proportions = (proportions * len(cls_idx)).astype(int)
        # Assign remainder to last client
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()

        start = 0
        for c in range(num_clients):
            end = start + proportions[c]
            client_indices[c].extend(cls_idx[start:end].tolist())
            start = end

    result = []
    for c in range(num_clients):
        idx = np.array(client_indices[c])
        rng.shuffle(idx)
        result.append((X[idx], y[idx]))

    sizes = [len(r[0]) for r in result]
    logger.info("Non-IID split (α=%.2f): %d clients, sizes=%s", alpha, num_clients, sizes)

    return result


def split_iid(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split data equally across clients in IID fashion.

    Args:
        X: Feature matrix of shape (N, D).
        y: Label vector of shape (N,).
        num_clients: Number of federated clients.
        seed: Random seed.

    Returns:
        List of (X_client, y_client) tuples, one per client.
    """
    rng = np.random.RandomState(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    splits = np.array_split(indices, num_clients)
    result = [(X[s], y[s]) for s in splits]

    sizes = [len(r[0]) for r in result]
    logger.info("IID split: %d clients, sizes=%s", num_clients, sizes)

    return result


# ──────────────────────────────────────────────────────────────────────
# Synthetic Data Fallback (preserved for testing)
# ──────────────────────────────────────────────────────────────────────

def generate_synthetic_data(
    num_samples: int = 1000,
    num_features: int = 122,
    anomaly_ratio: float = 0.3,
    num_nodes: int = 50,
    num_edges: int = 200,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Generate synthetic connection data and trust graph.

    Normal samples: N(0, 1).  Anomalous samples: N(3, 2).
    Trust graph: random edges, degree-correlated trust labels.

    Args:
        num_samples: Total number of connection records.
        num_features: Feature dimension per record.
        anomaly_ratio: Fraction of anomalous samples.
        num_nodes: Number of trust graph nodes.
        num_edges: Number of trust graph edges.
        seed: Random seed.

    Returns:
        ((X, y), (edge_index, node_features, trust_labels))
    """
    rng = np.random.RandomState(seed)
    num_anomaly = int(num_samples * anomaly_ratio)
    num_normal = num_samples - num_anomaly

    X_normal = rng.randn(num_normal, num_features).astype(np.float32)
    X_anomaly = (rng.randn(num_anomaly, num_features) * 2 + 3).astype(np.float32)
    X = np.concatenate([X_normal, X_anomaly], axis=0)
    y = np.concatenate([np.zeros(num_normal), np.ones(num_anomaly)]).astype(np.int64)

    # Shuffle
    perm = rng.permutation(num_samples)
    X, y = X[perm], y[perm]

    # Trust graph
    edge_index = rng.randint(0, num_nodes, size=(2, num_edges)).astype(np.int64)
    node_features = rng.randn(num_nodes, num_features).astype(np.float32) * 0.01

    degree = np.zeros(num_nodes)
    for i in range(num_edges):
        degree[edge_index[0, i]] += 1
        degree[edge_index[1, i]] += 1
    trust_labels = (degree > np.median(degree)).astype(np.float32)

    logger.info(
        "Synthetic data: %d samples (%d normal, %d anomaly), graph: %d nodes, %d edges",
        num_samples, num_normal, num_anomaly, num_nodes, num_edges,
    )

    return (X, y), (edge_index, node_features, trust_labels)


# ──────────────────────────────────────────────────────────────────────
# Module test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from ae_fzta import config

    # Test NSL-KDD
    X_tr, y_tr, X_te, y_te = load_nsl_kdd(
        config.NSL_TRAIN_PATH,
        config.NSL_TEST_PATH,
        config.NSL_COLUMN_NAMES,
        config.NSL_CATEGORICAL_COLS,
    )
    logger.info("NSL-KDD X_train: %s, y_train: %s", X_tr.shape, y_tr.shape)
    logger.info("NSL-KDD X_test: %s, y_test: %s", X_te.shape, y_te.shape)

    # Test TON_IoT
    X_ton, y_ton = load_ton_iot(
        config.TON_TRAIN_PATH,
        config.TON_DROP_COLS,
        config.TON_CATEGORICAL_COLS,
    )
    logger.info("TON_IoT X: %s, y: %s", X_ton.shape, y_ton.shape)

    # Test trust graph from TON_IoT (has actual IPs)
    df_ton = pd.read_csv(config.TON_TRAIN_PATH)
    df_ton.columns = [c.strip("\ufeff").strip() for c in df_ton.columns]
    edge_idx, node_feat, trust_lbl = build_trust_graph(
        df_ton, "src_ip", "dst_ip", "label", feature_dim=config.GNN_INPUT_DIM,
    )
    logger.info("Trust graph: edge_index=%s, node_feat=%s", edge_idx.shape, node_feat.shape)

    # Test non-IID split
    splits = split_non_iid(X_tr, y_tr, num_clients=3, alpha=0.5)
    for i, (xs, ys) in enumerate(splits):
        logger.info("Client %d: %d samples, attack_ratio=%.2f", i, len(xs), ys.mean())

    logger.info("✅ preprocessor.py all tests passed.")
