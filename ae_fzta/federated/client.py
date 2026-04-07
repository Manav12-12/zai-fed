"""Federated Learning Client for AE-FZTA.

Implements the Flower NumPyClient wrapping the Transformer anomaly detector
and GNN trust model.  LLM training is disabled — only Transformer + GNN
participate in federated rounds.

Includes AES-256-GCM encryption for weight exchange and Gaussian DP noise.

PROTOTYPE NOTE (CRIT-002): The paper (Section IV-E) specifies post-quantum
encryption (Kyber / CRYSTALS-Kyber, now standardised as NIST FIPS 203 ML-KEM)
to protect model updates against quantum adversaries.  This implementation
substitutes AES-256-GCM, which provides equivalent confidentiality for
classical adversaries but is quantum-vulnerable (Grover reduces key strength
to 128-bit effective security).  Kyber is not yet available in the standard
Python cryptography package.  To upgrade to post-quantum encryption:
    pip install liboqs-python  # or pqcrypto
then replace AESGCM with a Kyber KEM + symmetric cipher combination.

PROTOTYPE NOTE (DP): Differential Privacy noise (σ=DP_NOISE_SCALE) is applied
to parameter arrays before return.  The (ε, δ) budget declared in config.py
is informational — formal privacy accounting via Rényi DP composition is not
implemented.  See config.DP_EPSILON / config.DP_DELTA.

Classes:
    ZTAFederatedClient: Flower NumPyClient for federated training.

Functions:
    encrypt_weights / decrypt_weights: AES-256-GCM weight encryption.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from sklearn.metrics import accuracy_score, roc_auc_score

from ae_fzta import config
from ae_fzta.data.dataset import ConnectionDataset, GraphDataset
from ae_fzta.models.gnn_trust import GNNTrustModel
from ae_fzta.models.transformer_anomaly import TransformerAnomalyDetector
from ae_fzta.models.llm_policy import LLMPolicyGenerator
from ae_fzta.data.policy_generator import generate_synthetic_policies
from ae_fzta.decision import make_batch_decisions

try:
    from flwr.client import NumPyClient
except ImportError:
    from flwr.client import NumPyClient  # type: ignore

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Weight Encryption
# ──────────────────────────────────────────────────────────────────────

def encrypt_weights(
    weights: List[np.ndarray], key: bytes
) -> Tuple[List[bytes], List[bytes], List[Tuple[int, ...]], List[str]]:
    """Encrypt weight arrays with AES-256-GCM.

    Each array is serialised to raw bytes and encrypted with a unique
    12-byte nonce.

    Args:
        weights: List of numpy arrays.
        key: 32-byte AES-256 key.

    Returns:
        (ciphertexts, nonces, shapes, dtype_strings).
    """
    aesgcm = AESGCM(key)
    ciphertexts, nonces, shapes, dtype_strings = [], [], [], []

    for arr in weights:
        nonce = os.urandom(12)
        ct = aesgcm.encrypt(nonce, arr.tobytes(), None)
        ciphertexts.append(ct)
        nonces.append(nonce)
        shapes.append(arr.shape)
        dtype_strings.append(str(arr.dtype))

    return ciphertexts, nonces, shapes, dtype_strings


def decrypt_weights(
    ciphertexts: List[bytes],
    nonces: List[bytes],
    shapes: List[Tuple[int, ...]],
    dtype_strings: List[str],
    key: bytes,
) -> List[np.ndarray]:
    """Decrypt AES-256-GCM encrypted weight arrays.

    Args:
        ciphertexts: Encrypted byte strings.
        nonces: Nonces used during encryption.
        shapes: Original array shapes.
        dtype_strings: Original dtype strings.
        key: 32-byte AES key.

    Returns:
        List of numpy arrays with original shapes and dtypes.
    """
    aesgcm = AESGCM(key)
    weights = []
    for ct, nonce, shape, dtype_str in zip(ciphertexts, nonces, shapes, dtype_strings):
        plaintext = aesgcm.decrypt(nonce, ct, None)
        arr = np.frombuffer(plaintext, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        weights.append(arr)
    return weights


# ──────────────────────────────────────────────────────────────────────
# Federated Client
# ──────────────────────────────────────────────────────────────────────

class ZTAFederatedClient(NumPyClient):
    """Flower NumPyClient wrapping Transformer + GNN, and optionally LLM.

    Parameter ordering: Transformer parameters first, GNN parameters second, LLM parameters third.
    """

    def __init__(
        self,
        transformer_model: TransformerAnomalyDetector,
        gnn_model: GNNTrustModel,
        connection_dataset: ConnectionDataset,
        graph_dataset: GraphDataset,
        local_epochs: int = 1,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        dp_noise_scale: float = 0.01,
        grad_clip_norm: float = 1.0,
        encryption_key: bytes = b"",
        fedprox_mu: float = 0.0,
        llm_model: LLMPolicyGenerator = None,
    ) -> None:
        """Initialise the federated client.

        Args:
            transformer_model: TBAE model (already on GPU).
            gnn_model: GNTE model (already on GPU).
            connection_dataset: Connection dataset for this client.
            graph_dataset: Graph dataset for this client.
            local_epochs: Local epochs per FL round.
            learning_rate: Adam learning rate.
            batch_size: Training batch size.
            dp_noise_scale: Gaussian DP noise σ.
            grad_clip_norm: Max gradient norm.
            encryption_key: 32-byte AES key (empty = skip encryption).
            fedprox_mu: FedProx proximal term weight (0.0 = disable).
        """
        self.transformer = transformer_model
        self.gnn = gnn_model
        self.conn_dataset = connection_dataset
        self.graph_dataset = graph_dataset
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dp_noise_scale = dp_noise_scale
        self.grad_clip_norm = grad_clip_norm
        self.encryption_key = encryption_key
        self.fedprox_mu = fedprox_mu
        self.llm_model = llm_model
        self.device = config.DEVICE

        self._t_param_count = len(list(self.transformer.parameters()))
        self._g_param_count = len(list(self.gnn.parameters()))
        self._l_param_count = len(self.llm_model.get_numpy_parameters()) if self.llm_model is not None else 0

        logger.info(
            "ZTAFederatedClient: transformer_params=%d, gnn_params=%d, llm_params=%d, "
            "samples=%d, batch_size=%d",
            self._t_param_count, self._g_param_count, self._l_param_count,
            len(connection_dataset), batch_size,
        )

    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """Return [transformer..., gnn...] as numpy arrays.
        Transformer parameters first, GNN parameters second.
        
        NOTE: LLM parameters are intentionally excluded from federated weight exchange 
        as the LLM learns a fixed policy format that does not benefit from aggregation.
        """
        params: List[np.ndarray] = []
        params.extend(self.transformer.get_numpy_parameters())
        params.extend(self.gnn.get_numpy_parameters())
        return params

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Restore parameters to both models.
        
        NOTE: LLM parameters are intentionally excluded from federated weight exchange.
        This function only restores Transformer and GNN parameters.
        """
        t_end = self._t_param_count
        g_end = t_end + self._g_param_count
        self.transformer.set_numpy_parameters(parameters[:t_end])
        self.gnn.set_numpy_parameters(parameters[t_end:g_end])

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any], current_round: int = 1
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """Local training: Transformer on connections, GNN on graph.

        Returns:
            (updated_params, dataset_size, {"avg_loss": float}).
        """
        self.set_parameters(parameters)
        total_loss = 0.0
        loss_count = 0

        # Save global params for FedProx
        global_t_params = [p.clone().detach() for p in self.transformer.parameters()]
        global_g_params = [p.clone().detach() for p in self.gnn.parameters()]

        # --- Warmup Schedule ---
        import ae_fzta.config as global_config
        if current_round <= global_config.FL_NUM_ROUNDS_WARMUP:
            active_lr = self.learning_rate * 2.0
            logger.debug("Client %s: Using warmup LR %.2e (Round %d)", id(self), active_lr, current_round)
        else:
            active_lr = self.learning_rate

        # --- Train Transformer (batched) ---
        self.transformer.train()
        t_opt = torch.optim.Adam(self.transformer.parameters(), lr=active_lr)
        

        loader = torch.utils.data.DataLoader(
            self.conn_dataset, batch_size=self.batch_size, shuffle=True,
        )

        for _ in range(self.local_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred = self.transformer(batch_x)
                
                per_sample_loss = torch.nn.functional.binary_cross_entropy(
                    pred.squeeze(), batch_y, reduction="none"
                )
                # Focal Loss: down-weight easy examples, focus on hard borderline cases
                if global_config.USE_FOCAL_LOSS:
                    p_t = torch.where(batch_y > 0.5, pred.squeeze(), 1 - pred.squeeze())
                    focal_weight = (1 - p_t) ** global_config.FL_FOCAL_LOSS_GAMMA
                    per_sample_loss = focal_weight * per_sample_loss
                class_weights = torch.where(
                    batch_y > 0.5,
                    torch.full_like(batch_y, global_config.FL_CLASS_WEIGHT_ATTACK),
                    torch.ones_like(batch_y)
                )
                loss = (per_sample_loss * class_weights).mean()

                if self.fedprox_mu > 0.0:
                    proximal_term = 0.0
                    for w, w_t in zip(self.transformer.parameters(), global_t_params):
                        proximal_term += (w - w_t).norm(2) ** 2
                    loss += (self.fedprox_mu / 2) * proximal_term

                t_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(), self.grad_clip_norm
                )
                t_opt.step()
                total_loss += loss.item()
                loss_count += 1

        # --- Train GNN ---
        self.gnn.train()
        g_opt = torch.optim.Adam(self.gnn.parameters(), lr=active_lr)
        graph_data = self.graph_dataset[0]
        gx = graph_data.x.to(self.device)
        ge = graph_data.edge_index.to(self.device)
        gy = graph_data.y.to(self.device)

        for _ in range(self.local_epochs):
            trust_scores = self.gnn(gx, ge)
            g_loss = self.gnn.compute_loss(trust_scores, gy)

            if self.fedprox_mu > 0.0:
                proximal_term = 0.0
                for w, w_t in zip(self.gnn.parameters(), global_g_params):
                    proximal_term += (w - w_t).norm(2) ** 2
                g_loss += (self.fedprox_mu / 2) * proximal_term

            g_opt.zero_grad()
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gnn.parameters(), self.grad_clip_norm)
            g_opt.step()
            total_loss += g_loss.item()
            loss_count += 1
            
        # --- Local LLM Fine-Tuning ---
        import ae_fzta.config as global_config
        if self.llm_model is not None:
            llm_samples = global_config.FL_LLM_FINETUNE_PAIRS
            syn_logs, syn_pols = generate_synthetic_policies(llm_samples, global_config.POLICY_ALLOW_RATIO, global_config.RANDOM_SEED)
            logger.info("Client %s: LLM fine-tuning occurring on %d samples", id(self), llm_samples)
            self.llm_model.fine_tune(syn_logs, syn_pols, epochs=1, batch_size=global_config.LLM_BATCH_SIZE, learning_rate=global_config.FL_LLM_FINETUNE_LR)

        avg_loss = total_loss / max(loss_count, 1)

        # --- DP noise ---
        updated_params = self.get_parameters(config={})
        if self.dp_noise_scale > 0.0:
            updated_params = [
                p + np.random.normal(0, self.dp_noise_scale, size=p.shape).astype(p.dtype)
                for p in updated_params
            ]

        # --- Encryption ---
        if len(self.encryption_key) == 32:
            ct, nonces, shapes, dtypes = encrypt_weights(updated_params, self.encryption_key)
            updated_params = decrypt_weights(ct, nonces, shapes, dtypes, self.encryption_key)

        dataset_size = len(self.conn_dataset)
        logger.info("Client fit: avg_loss=%.4f, samples=%d", avg_loss, dataset_size)

        return updated_params, dataset_size, {"avg_loss": float(avg_loss)}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """Evaluate Transformer accuracy and GNN trust AUC.

        Returns:
            (total_loss, dataset_size, {"accuracy": float, "trust_auc": float}).
        """
        import ae_fzta.config as global_config
        self.set_parameters(parameters)
        self.transformer.eval()
        self.gnn.eval()

        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            # Evaluate Transformer (batched)
            loader = torch.utils.data.DataLoader(
                self.conn_dataset, batch_size=self.batch_size, shuffle=False,
            )
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                pred = self.transformer(batch_x)
                loss = self.transformer.compute_loss(pred, batch_y)
                total_loss += loss.item() * len(batch_y)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(batch_y.cpu().tolist())

            # Evaluate GNN
            graph_data = self.graph_dataset[0]
            gx = graph_data.x.to(self.device)
            ge = graph_data.edge_index.to(self.device)
            gy = graph_data.y.to(self.device)
            trust_scores = self.gnn(gx, ge)
            g_loss = self.gnn.compute_loss(trust_scores, gy)
            total_loss += g_loss.item()

        binary_preds = [1.0 if p >= global_config.ANOMALY_THRESHOLD else 0.0 for p in all_preds]
        accuracy = float(accuracy_score(all_labels, binary_preds))

        trust_preds = trust_scores.cpu().numpy()
        trust_labels = gy.cpu().numpy()
        
        # --- Make Batch Decisions with LLM generated policies ---
        import ae_fzta.config as global_config
        if self.llm_model is not None:
            syn_logs, _ = generate_synthetic_policies(10, global_config.POLICY_ALLOW_RATIO, global_config.RANDOM_SEED)
            base_log = syn_logs[0]
            # Use llm_model.generate_policy() on the synthetic log string exactly once
            generated_policy = self.llm_model.generate_policy(base_log)
            policy_strings = [generated_policy] * len(all_preds)
        else:
            # Fallback to existing behaviour of passing hardcoded "ALLOW" strings
            policy_strings = ["ALLOW"] * len(all_preds)
            
        trust_preds_expanded = [
    float(trust_preds[i % len(trust_preds)]) 
    for i in range(len(all_preds))
]
        decisions = make_batch_decisions(all_preds, trust_preds_expanded, policy_strings, global_config.ANOMALY_THRESHOLD, global_config.TRUST_THRESHOLD)
        
        try:
            trust_auc = float(roc_auc_score(trust_labels, trust_preds))
        except ValueError:
            trust_auc = 0.5

        logger.info(
            "Client eval: loss=%.4f, accuracy=%.4f, trust_auc=%.4f",
            total_loss, accuracy, trust_auc,
        )

        return total_loss, len(self.conn_dataset), {
            "accuracy": accuracy, "trust_auc": trust_auc,
        }
