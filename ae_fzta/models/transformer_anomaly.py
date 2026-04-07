"""Transformer-Based Behavioural Analysis Engine (TBAE).

Implements the Transformer-based anomaly detection model described in
Section IV-B of the AE-FZTA paper (Paper Equations 5–8). Models sequential
user/device behaviour to detect anomalous access sessions in real time.

Classes:
    PositionalEncoding: Sinusoidal positional encoding module.
    TransformerAnomalyDetector: Full TBAE model with attention-based
        sequence classification and anomaly scoring.
"""

import logging
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer inputs.

    Adds position-dependent signals to input embeddings so the Transformer
    can learn order-sensitive representations. Uses the formulation from
    "Attention Is All You Need" (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        """Initialise positional encoding.

        Args:
            d_model: Model embedding dimension.
            max_len: Maximum supported sequence length.
            dropout: Dropout rate applied after adding positional encoding.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings: shape (1, max_len, d_model)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerAnomalyDetector(nn.Module):
    """Transformer-Based Behavioural Analysis Engine (TBAE).

    Implements Paper Equations 5–8 for sequential anomaly detection.

    Architecture:
        1. Linear projection: input_dim → d_model
        2. Positional encoding
        3. Transformer encoder (multi-head self-attention, Eq. 6)
        4. Mean pooling over sequence dimension
        5. Classification head → sigmoid → anomaly score in [0, 1]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ) -> None:
        """Initialise the TBAE model.

        Args:
            input_dim: Raw feature dimension per timestep (x_t ∈ R^d, Eq. 5).
            d_model: Internal model dimension (d_k in Eq. 6). Must be
                divisible by nhead.
            nhead: Number of attention heads (Eq. 6).
            num_layers: Number of Transformer encoder layers.
            dim_feedforward: Hidden dimension of feed-forward sub-layers (Eq. 7).
            dropout: Dropout rate.
        """
        super().__init__()

        # Project raw features to model dimension
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head: pooled representation → anomaly score
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
        self.sigmoid = nn.Sigmoid()

        logger.info(
            "TransformerAnomalyDetector: input_dim=%d, d_model=%d, nhead=%d, "
            "layers=%d, ff=%d, dropout=%.2f",
            input_dim,
            d_model,
            nhead,
            num_layers,
            dim_feedforward,
            dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for a batch of sessions.

        Implements the forward pass corresponding to Eqs. 5–7:
            1. Project input features (Eq. 5 representation).
            2. Add positional encoding.
            3. Apply multi-head self-attention (Eq. 6).
            4. Feed-forward + LayerNorm (Eq. 7).
            5. Mean-pool → classify → sigmoid.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Anomaly scores of shape (batch_size,) in range [0, 1].
        """
        # (batch, seq, input_dim) → (batch, seq, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder: (batch, seq, d_model) → (batch, seq, d_model)
        x = self.transformer_encoder(x)

        # Mean pooling over sequence dimension: (batch, d_model)
        x = x.mean(dim=1)

        # Classification head: (batch, d_model) → (batch, 1) → (batch,)
        x = self.classifier(x).squeeze(-1)
        x = self.sigmoid(x)
        return x

    def compute_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute anomaly loss (WARN-001: intentional deviation from Paper Eq. 8).

        Paper Eq. 8 specifies MSE: L_anomaly = (1/N) * Σ(y_i - y'_i)²
        This implementation uses Binary Cross-Entropy (BCE) instead.

        Rationale: MSE is suboptimal for binary classification because it
        produces poorly calibrated probability scores near 0 and 1 (flat
        gradients).  BCE directly optimises the log-likelihood of the binary
        label, producing sharper decision boundaries and faster convergence.
        The output anomaly score remains a sigmoid scalar in [0, 1] per
        the paper's formulation — only the training objective changes.

        Args:
            predictions: Predicted anomaly scores, shape (batch_size,).
            targets: Ground truth labels (0 or 1), shape (batch_size,).

        Returns:
            Scalar BCE loss tensor.
        """
        return torch.nn.functional.binary_cross_entropy(predictions, targets)

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
    model = TransformerAnomalyDetector(
        input_dim=16,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )

    # Set eval mode for deterministic testing (dropout disabled)
    model.eval()

    # Test input: batch=4, seq_len=20, features=16
    x = torch.randn(4, 20, 16)
    with torch.no_grad():
        scores = model(x)

    # Verify output shape
    assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"

    # Verify output bounded in [0, 1]
    assert torch.all(scores >= 0.0) and torch.all(scores <= 1.0), (
        f"Scores out of range: min={scores.min():.4f}, max={scores.max():.4f}"
    )

    # Verify loss is a finite scalar
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = model.compute_loss(scores, targets)
    assert loss.dim() == 0, f"Loss should be scalar, got dim={loss.dim()}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"

    # Verify parameter round-trip (eval mode, no_grad — deterministic)
    params = model.get_numpy_parameters()
    assert len(params) > 0, "No parameters extracted"
    model.set_numpy_parameters(params)
    with torch.no_grad():
        scores_after = model(x)
    assert torch.allclose(scores, scores_after, atol=1e-5), "Parameter round-trip failed"

    logger.info("Scores: %s", scores.detach().numpy())
    logger.info("Loss: %.6f", loss.item())
    logger.info("✅ STEP 5 COMPLETE — transformer_anomaly.py all tests passed.")
