"""PyTorch Dataset classes for AE-FZTA.

Classes:
    ConnectionDataset: Dataset wrapping per-connection feature arrays + labels.
    GraphDataset: Dataset wrapping a single trust graph.
"""

import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class ConnectionDataset(Dataset):
    """PyTorch Dataset for individual network connection records.

    Each item is a (feature_tensor, label_tensor) pair.  The Transformer
    model expects shape (batch, seq_len, features), so ``__getitem__``
    returns features with an extra length-1 sequence dimension:
    shape (1, num_features).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialise from feature matrix and label vector.

        Args:
            X: Feature matrix of shape (N, D), dtype float32.
            y: Label vector of shape (N,), dtype int64.

        Raises:
            ValueError: If X and y have different sample counts.
        """
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) must have equal length.")
        if len(X) == 0:
            raise ValueError("Cannot create ConnectionDataset from empty data.")

        self._X = X
        self._y = y
        logger.info(
            "ConnectionDataset created: %d samples, %d features",
            len(X), X.shape[1],
        )

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (feature_tensor, label_tensor).

        Features are returned with shape (1, D) — a length-1 sequence
        for the Transformer encoder.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (features [1, D], label scalar), both float32.
        """
        features = torch.tensor(self._X[idx], dtype=torch.float32).unsqueeze(0)  # (1, D)
        label = torch.tensor(float(self._y[idx]), dtype=torch.float32)
        return features, label

    def __repr__(self) -> str:
        return f"ConnectionDataset(size={len(self)}, features={self._X.shape[1]})"


class GraphDataset(Dataset):
    """PyTorch Dataset wrapping a single trust graph.

    Always has length 1.  Returns the same Data object for any index.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        node_features: np.ndarray,
        trust_labels: np.ndarray,
    ) -> None:
        """Initialise with graph arrays and convert to a Data object.

        Args:
            edge_index: Shape (2, num_edges), dtype int64.
            node_features: Shape (num_nodes, feature_dim), dtype float32.
            trust_labels: Shape (num_nodes,), dtype float32.
        """
        self._data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(trust_labels, dtype=torch.float32),
        )
        logger.info(
            "GraphDataset created: %d nodes, %d edges",
            node_features.shape[0], edge_index.shape[1],
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Data:
        return self._data

    def __repr__(self) -> str:
        return (
            f"GraphDataset(nodes={self._data.x.shape[0]}, "
            f"edges={self._data.edge_index.shape[1]})"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test with synthetic data
    rng = np.random.RandomState(42)
    X = rng.randn(100, 122).astype(np.float32)
    y = rng.randint(0, 2, 100).astype(np.int64)

    ds = ConnectionDataset(X, y)
    assert len(ds) == 100
    feat, lbl = ds[0]
    assert feat.shape == (1, 122), f"Expected (1, 122), got {feat.shape}"
    assert lbl.shape == ()

    # Graph
    edge_idx = rng.randint(0, 20, (2, 50)).astype(np.int64)
    node_feat = rng.randn(20, 122).astype(np.float32)
    trust_lbl = rng.randint(0, 2, 20).astype(np.float32)
    gd = GraphDataset(edge_idx, node_feat, trust_lbl)
    assert len(gd) == 1
    data = gd[0]
    assert data.x.shape == (20, 122)

    logger.info("✅ dataset.py all tests passed.")
