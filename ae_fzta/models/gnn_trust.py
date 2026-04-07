"""Graph Neural Trust Engine (GNTE).

Implements the GNN-based trust propagation model described in Section IV-C
of the AE-FZTA paper (Paper Equations 9–12). Constructs dynamic interaction
graphs and infers contextual trust scores for users, devices, and services.

Classes:
    GNNTrustModel: Graph Convolutional Network for trust scoring.
"""

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)


class GNNTrustModel(nn.Module):
    """Graph Neural Trust Engine using stacked GCN layers.

    Implements Paper Equations 9–12 for trust propagation and scoring.

    Architecture:
        1. Stack of GNN_NUM_LAYERS GCNConv layers (Eq. 10).
        2. ReLU activation + dropout between layers (except last).
        3. Linear trust head → sigmoid → trust score in [0, 1] (Eq. 11).
        4. BCE loss for trust label prediction (Eq. 12).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the GNN trust model.

        Args:
            input_dim: Node feature dimension (h_i^0 ∈ R^d).
            hidden_dim: Hidden layer dimension (W^(l) in Eq. 10).
            num_layers: Number of GCN layers to stack (L in Eq. 11).
            dropout: Dropout rate between layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # Build GCN layers using nn.ModuleList — never hardcode depth
        self.convs = nn.ModuleList()
        # First layer: input_dim → hidden_dim
        self.convs.append(GCNConv(input_dim, hidden_dim))
        # Intermediate layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Trust scoring head: W_trust^T h_i^(L) → sigmoid (Eq. 11)
        self.trust_head = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        logger.info(
            "GNNTrustModel: input_dim=%d, hidden_dim=%d, layers=%d, dropout=%.2f",
            input_dim,
            hidden_dim,
            num_layers,
            dropout,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute trust scores for all nodes.

        Implements Eq. 10 (message passing) followed by Eq. 11 (trust scoring).

        Args:
            x: Node feature matrix of shape (num_nodes, input_dim).
            edge_index: Edge index tensor of shape (2, num_edges), dtype long.

        Returns:
            Trust scores of shape (num_nodes,) in range [0, 1].
        """
        # Apply GCN layers with ReLU + dropout (except after last conv)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Trust head: (num_nodes, hidden_dim) → (num_nodes, 1) → (num_nodes,)
        trust_scores = self.trust_head(x).squeeze(-1)
        trust_scores = self.sigmoid(trust_scores)
        return trust_scores

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute binary cross-entropy loss per Paper Eq. 12.

        L_trust = -Σ [t_i log(T(v_i)) + (1-t_i) log(1-T(v_i))]

        Uses BCE directly on sigmoid outputs (not BCEWithLogitsLoss).

        Args:
            predictions: Predicted trust scores in [0, 1], shape (num_nodes,).
            targets: Ground truth trust labels (0 or 1), shape (num_nodes,).

        Returns:
            Scalar BCE loss tensor.
        """
        return F.binary_cross_entropy(predictions, targets)

    def get_numpy_parameters(self) -> List[np.ndarray]:
        """Extract model parameters as a list of numpy arrays.

        Returns:
            List of numpy arrays, one per parameter tensor, in the order
            returned by self.parameters().
        """
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_numpy_parameters(self, params: List[np.ndarray]) -> None:
        """Restore model parameters from numpy arrays.

        Args:
            params: List of numpy arrays matching model parameter shapes
                and ordering from get_numpy_parameters().
        """
        for param, np_arr in zip(self.parameters(), params):
            param.data = torch.tensor(np_arr, dtype=param.dtype, device=param.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create model with test dimensions
    model = GNNTrustModel(
        input_dim=16,
        hidden_dim=32,
        num_layers=3,
        dropout=0.1,
    )

    # Synthetic graph: 30 nodes, 80 edges
    num_nodes = 30
    x = torch.randn(num_nodes, 16)
    edge_index = torch.randint(0, num_nodes, (2, 80))

    # Forward pass
    model.eval()
    scores = model(x, edge_index)

    # Verify output shape matches number of nodes
    assert scores.shape == (num_nodes,), f"Expected shape ({num_nodes},), got {scores.shape}"

    # Verify all output values in [0, 1]
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0), (
        f"Scores out of range: min={scores.min():.4f}, max={scores.max():.4f}"
    )

    # Verify loss
    targets = torch.randint(0, 2, (num_nodes,)).float()
    loss = model.compute_loss(scores, targets)
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # Verify parameter round-trip
    params = model.get_numpy_parameters()
    assert len(params) > 0, "No parameters extracted"
    model.set_numpy_parameters(params)
    scores_after = model(x, edge_index)
    assert torch.allclose(scores, scores_after, atol=1e-6), "Parameter round-trip failed"

    logger.info("Trust scores sample: %s", scores[:5].detach().numpy())
    logger.info("Loss: %.6f", loss.item())
    logger.info("✅ STEP 6 COMPLETE — gnn_trust.py all tests passed.")
