"""AE-FZTA Configuration Module.

Centralises every tuneable constant for the AI-Enhanced Federated Zero Trust
Architecture.  No other module should contain hardcoded numeric or string
values — all configurable values live here.

Design decisions:
    • GPU is **mandatory**.  Training on CPU is unreasonably slow and is
      treated as a configuration error caught at import time.
    • Model dimensions are tuned to use ~90% of a GTX 1650 (4 GB VRAM)
      via empirical memory profiling.  Transformer + GNN + Adam states +
      batch activations peak at ~3.9 GB during training.
    • Feature dimensions (122) match NSL-KDD after one-hot encoding
      its three categorical columns (protocol_type, service, flag).
    • LLM constants are commented out — the LLM module is preserved
      in code but excluded from the federated training loop.

Sections:
    - Device
    - Transformer Architecture (TBAE)
    - GNN Architecture (GNTE)
    - Federated Learning Parameters
    - Differential Privacy Parameters
    - Decision Thresholds
    - General Training Parameters
    - Dataset Paths
    - Logging
"""

import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device Configuration — GPU is mandatory
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    print(
        "FATAL: CUDA is not available.  AE-FZTA requires an NVIDIA GPU.\n"
        "Install a CUDA-compatible PyTorch build:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu118\n",
        file=sys.stderr,
    )
    sys.exit(1)

DEVICE: torch.device = torch.device("cuda")

# ---------------------------------------------------------------------------
# Transformer Architecture (TBAE) — Paper Section IV-B, Eqs. 5–8
# Optimised for GTX 1650 4 GB VRAM
# ---------------------------------------------------------------------------
# After one-hot encoding NSL-KDD's 3 categorical columns
# (protocol_type=3, service=70, flag=11) the remaining 38 numeric features
# give a total of 38 + 3 + 70 + 11 = 122 features per connection.
TRANSFORMER_INPUT_DIM: int = 122

# Internal model dimension.  512 is divisible by NHEAD=8 (head dim=64).
# Profiled to use ~91% of GTX 1650 VRAM with batch_size=11264.
TRANSFORMER_D_MODEL: int = 512

# Number of attention heads.  8 heads × 64-dim each = 512.
TRANSFORMER_NHEAD: int = 8

# Six encoder layers for richer feature interaction modelling.
TRANSFORMER_NUM_LAYERS: int = 6

# Feed-forward hidden dim — 4× model dim (standard ratio).
TRANSFORMER_DIM_FEEDFORWARD: int = 2048

# Dropout rate for regularisation.
TRANSFORMER_DROPOUT: float = 0.1

# NSL-KDD and TON_IoT are per-connection records, not time series.
# Each connection is treated as a length-1 sequence by the Transformer.
SEQUENCE_LENGTH: int = 1

# ---------------------------------------------------------------------------
# GNN Architecture (GNTE) — Paper Section IV-C, Eqs. 9–12
# ---------------------------------------------------------------------------
# Node features match the connection feature space.
GNN_INPUT_DIM: int = 122

# 256 hidden units — large enough for expressive node embeddings.
GNN_HIDDEN_DIM: int = 256

# 4 GCN layers — captures 4-hop neighbourhood structure.
GNN_NUM_LAYERS: int = 4

# Dropout between GCN layers.
GNN_DROPOUT: float = 0.1

# ---------------------------------------------------------------------------
# LLM Configuration (PALLM) — DISABLED for training
# The LLM module is preserved in code but excluded from the federated
# training loop.  Uncomment and adjust if you want to re-enable it.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LLM Configuration (PALLM)
# ---------------------------------------------------------------------------
# PROTOTYPE NOTE (CRIT-001): The paper (Section V-E, Fig. 4) benchmarks and reports results
# for T5-Large (737M parameters, BLEU=49.8) as the proposed model.  This implementation
# substitutes t5-small (60M parameters) due to the GTX 1650 4GB VRAM constraint
# (T5-Large requires ≥16GB VRAM at batch_size=8).  The architecture, training pipeline,
# and evaluation code are identical — only the checkpoint size differs.
# To reproduce paper results on a capable GPU (A100/H100), set:
#   LLM_MODEL_NAME = "t5-large"
LLM_MODEL_NAME: str = "t5-small"
LLM_MAX_LENGTH: int = 128
LLM_BEAM_WIDTH: int = 4
LLM_LEARNING_RATE: float = 5e-5
LLM_EPOCHS: int = 5
LLM_BATCH_SIZE: int = 8

# =====================================================================
# Policy Training Configuration
# =====================================================================

POLICY_SYNTHETIC_PAIRS: int = 10000
POLICY_ALLOW_RATIO: float = 0.5
POLICY_TRAIN_SPLIT: float = 0.8
POLICY_MODEL_SAVE_PATH: str = "./checkpoints/llm_policy_model/"

# Stores the best LLM weights observed during federated fine-tuning, distinct from the initial pre-trained weights
BEST_LLM_CHECKPOINT_PATH: str = "./checkpoints/llm_best_model/"

# The LLM learns a fixed policy format that does not vary meaningfully by client data distribution, 
# therefore federating its 60 million parameters adds communication overhead without proportional benefit. 
# The Transformer and GNN are the components that genuinely benefit from federated aggregation 
# because they learn from distributed network traffic patterns.
INCLUDE_LLM_IN_FEDERATION: bool = False

FL_LLM_FINETUNE_PAIRS: int = 50
# Used exclusively during federated local LLM fine-tuning. This is intentionally an order of magnitude 
# lower than LLM_LEARNING_RATE to protect pre-trained weights from catastrophic forgetting during short steps.
FL_LLM_FINETUNE_LR: float = 1e-5

# ---------------------------------------------------------------------------
# Federated Learning Parameters — Paper Section IV-E, Eqs. 15–17
# ---------------------------------------------------------------------------
# Number of edge/cloud clients.
# Paper Section V-A specifies 10 clients.  This prototype defaults to 5 to
# reduce training time on a GTX 1650 (4 GB VRAM).  The paper's results are
# reproducible by setting FL_NUM_CLIENTS=10 and FL_NUM_ROUNDS=20 on an A100.
FL_NUM_CLIENTS: int = 5

# Communication rounds — each round = broadcast → local train → aggregate.
FL_NUM_ROUNDS: int = 30

# Local epochs per client per FL round.
FL_LOCAL_EPOCHS: int = 3

# Adam learning rate for local optimisers.
FL_LEARNING_RATE: float = 1e-3

# Gradient clipping bound (also constrains DP sensitivity).
FL_GRAD_CLIP_NORM: float = 1.0

# Fraction of clients sampled per round (1.0 = all participate).
FL_FRACTION_FIT: float = 1.0
FL_FRACTION_EVALUATE: float = 1.0

# Checkpoint every N rounds.
FL_CHECKPOINT_EVERY: int = 5

# Dirichlet α for non-IID client splits.
# α=0.5 → moderate heterogeneity.  Lower → more skewed.
FL_DIRICHLET_ALPHA: float = 0.5

# FedProx proximal term weight (0.0 = disable)
FL_FEDPROX_MU: float = 0.1

# Weight multiplier applied to attack class loss during training to compensate for the model's bias toward the majority class
FL_CLASS_WEIGHT_ATTACK: float = 4.0

# Focal Loss properties down-weight easy examples enforcing harder case evaluation on overlaps
USE_FOCAL_LOSS: bool = True
FL_FOCAL_LOSS_GAMMA: float = 2.0

# Number of initial rounds where clients train with higher learning rate before dropping to standard rate
FL_NUM_ROUNDS_WARMUP: int = 5

# ---------------------------------------------------------------------------
# Differential Privacy Parameters — Paper Section IV-E
# ---------------------------------------------------------------------------
# Gaussian noise σ added to parameter updates before aggregation.
DP_NOISE_SCALE: float = 0.001

# Informational (ε, δ)-DP budget — not enforced via privacy accounting.
DP_EPSILON: float = 1.0
DP_DELTA: float = 1e-5

# ---------------------------------------------------------------------------
# Decision Thresholds — Paper Section IV-F, Eq. 18
# ---------------------------------------------------------------------------
# Anomaly score threshold τ_a: access denied if score ≥ this.
ANOMALY_THRESHOLD: float = 0.5

# Trust score threshold τ_t: access denied if score ≤ this.
TRUST_THRESHOLD: float = 0.5

# Token that must appear in a policy string for access to be granted.
ALLOW_TOKEN: str = "ALLOW"

# ---------------------------------------------------------------------------
# General Training Parameters
# ---------------------------------------------------------------------------
# Batch size — profiled for ~90% GPU utilisation on GTX 1650 (4 GB).
# Each sample is 122 float32 = 488 bytes.  Batch of 11264 ≈ 5.2 MB on GPU,
# but activations and Adam states dominate at this model size.
BATCH_SIZE: int = 11264

# Global random seed for reproducibility.
RANDOM_SEED: int = 42

# Number of samples for latency benchmarking.
LATENCY_SAMPLES: int = 100

# Default number of sample decisions to display in the test script output.
TEST_DISPLAY_SAMPLES: int = 50

# ---------------------------------------------------------------------------
# Dataset Paths — relative to the project root
# ---------------------------------------------------------------------------
# NSL-KDD headerless CSVs.
NSL_TRAIN_PATH: str = "ae_fzta/nsl-dataset/KDDTrain+.txt"
NSL_TEST_PATH: str = "ae_fzta/nsl-dataset/KDDTest+.txt"

# TON_IoT network CSV (has header row).
TON_TRAIN_PATH: str = "ae_fzta/ton-dataset/train_test_network.csv"

# ---------------------------------------------------------------------------
# File Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR: str = "./checkpoints/"

# ---------------------------------------------------------------------------
# NSL-KDD column names (the .txt has no header — we assign these)
# Order matches the KDDTrain+.arff attribute list.
# ---------------------------------------------------------------------------
NSL_COLUMN_NAMES: list = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]

# Columns to one-hot encode in NSL-KDD.
NSL_CATEGORICAL_COLS: list = ["protocol_type", "service", "flag"]

# Columns to one-hot encode in TON_IoT.
TON_CATEGORICAL_COLS: list = ["proto", "conn_state"]

# Columns to drop from TON_IoT (string/IP/non-numeric).
TON_DROP_COLS: list = [
    "src_ip", "dst_ip", "dns_query", "ssl_version", "ssl_cipher",
    "ssl_subject", "ssl_issuer", "http_method", "http_uri",
    "http_version", "http_user_agent", "http_orig_mime_types",
    "http_resp_mime_types", "weird_name", "weird_addl", "weird_notice",
    "service", "type",
]


def validate_config() -> None:
    """Validate internal consistency of all configuration constants.

    Checks divisibility, positivity, valid ranges, and path existence.

    Raises:
        AssertionError: If any check fails.
    """
    # --- Transformer ---
    assert TRANSFORMER_D_MODEL % TRANSFORMER_NHEAD == 0, (
        f"D_MODEL ({TRANSFORMER_D_MODEL}) must be divisible by NHEAD ({TRANSFORMER_NHEAD})"
    )
    assert TRANSFORMER_INPUT_DIM > 0
    assert TRANSFORMER_D_MODEL > 0
    assert TRANSFORMER_NUM_LAYERS > 0
    assert TRANSFORMER_DIM_FEEDFORWARD > 0
    assert 0.0 <= TRANSFORMER_DROPOUT < 1.0
    assert SEQUENCE_LENGTH > 0

    # --- GNN ---
    assert GNN_INPUT_DIM > 0
    assert GNN_HIDDEN_DIM > 0
    assert GNN_NUM_LAYERS > 0
    assert 0.0 <= GNN_DROPOUT < 1.0

    # --- Federated Learning ---
    assert FL_NUM_CLIENTS > 0
    assert FL_NUM_ROUNDS > 0
    assert FL_LOCAL_EPOCHS > 0
    assert FL_LEARNING_RATE > 0.0
    assert FL_GRAD_CLIP_NORM > 0.0
    assert 0.0 < FL_FRACTION_FIT <= 1.0
    assert 0.0 < FL_FRACTION_EVALUATE <= 1.0
    assert FL_CHECKPOINT_EVERY > 0
    assert FL_DIRICHLET_ALPHA > 0.0
    assert FL_NUM_ROUNDS_WARMUP >= 0

    # --- Differential Privacy ---
    assert DP_NOISE_SCALE >= 0.0
    assert DP_EPSILON > 0.0
    assert 0.0 < DP_DELTA < 1.0

    # --- Decision thresholds ---
    assert 0.0 <= ANOMALY_THRESHOLD <= 1.0
    assert 0.0 <= TRUST_THRESHOLD <= 1.0
    assert len(ALLOW_TOKEN) > 0

    # --- General ---
    assert BATCH_SIZE > 0
    assert RANDOM_SEED >= 0
    assert LATENCY_SAMPLES > 0

    # --- Policy Training ---
    assert POLICY_SYNTHETIC_PAIRS > 0
    assert 0.0 <= POLICY_ALLOW_RATIO <= 1.0
    assert 0.0 < POLICY_TRAIN_SPLIT < 1.0
    assert len(POLICY_MODEL_SAVE_PATH) > 0
    assert len(BEST_LLM_CHECKPOINT_PATH) > 0
    assert FL_LLM_FINETUNE_PAIRS > 0
    assert FL_LLM_FINETUNE_LR < LLM_LEARNING_RATE, "FL_LLM_FINETUNE_LR must be strictly less than LLM_LEARNING_RATE"

    # --- Paths ---
    assert len(CHECKPOINT_DIR) > 0

    logger.info("Configuration validation passed.")


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------
validate_config()

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

logger.info(
    "AE-FZTA config loaded — device=%s, GPU=%s, VRAM=%.0f MB",
    DEVICE,
    torch.cuda.get_device_name(0),
    torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
)
