# AE-FZTA Setup Guide

Complete guide for setting up, training, and testing the AI-Enhanced Federated Zero Trust Architecture.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Datasets](#3-datasets)
4. [Training](#4-training)
5. [Testing a Trained Model](#5-testing-a-trained-model)
6. [LLM Policy Model](#6-llm-policy-model)
7. [Visualization](#7-visualization)
8. [Module Tests](#8-module-tests)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. System Requirements

| Requirement | Value | Notes |
|---|---|---|
| **Python** | 3.10+ | Type hints, match statements, torch compatibility |
| **NVIDIA GPU** | ≥4 GB VRAM | **Mandatory**. The project exits on startup if CUDA is unavailable. |
| **CUDA** | 12.x+ | Must match PyTorch CUDA wheel version |
| **RAM** | 8 GB+ | Pandas loading temporarily doubles dataset memory usage |
| **Disk** | ~1 GB | venv (~800 MB), datasets (~50 MB), checkpoints (~20 MB) |

### Why GPU is mandatory

`config.py` calls `sys.exit(1)` if CUDA is unavailable. This is intentional:

- NSL-KDD has 125,973 training rows. At batch_size=11264, GPU processes each batch in ~2ms vs ~200ms on CPU.
- Federated training multiplies this: 20 clients × 40 rounds × 2 epochs = 1,600 local training passes.
- **Total estimate: ~5 minutes on GPU vs ~4+ hours on CPU.**

---

## 2. Environment Setup

### Step 1: Create virtual environment

```bash
cd zai-fed
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install PyTorch with CUDA

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

The CUDA 12.4 wheel is forward-compatible with newer drivers (e.g. CUDA 13.x).

### Step 3: Install the project (editable mode)

```bash
pip install -e .
```

This registers `ae_fzta` as an installed Python package via symlink. After this, `import ae_fzta` works from anywhere — no `PYTHONPATH` needed. The `-e` flag means code changes take effect immediately without re-installing.

### Step 4: Install matplotlib

```bash
pip install matplotlib
```

Required for training result plots and confusion matrix generation.

### Step 5: Verify installation

```bash
python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
python3 -c "from ae_fzta.config import validate_config; print('Config OK')"
```

---

## 3. Datasets

AE-FZTA uses two complementary datasets:

| Dataset | Purpose | Used By | Location |
|---|---|---|---|
| **NSL-KDD** | Binary anomaly detection | Transformer (TBAE) | `ae_fzta/nsl-dataset/` |
| **TON_IoT** | Trust graph construction | GNN (GNTE) | `ae_fzta/ton-dataset/` |

### NSL-KDD

- **Train**: `KDDTrain+.txt` — 125,973 connections, 41 raw features → 122 after one-hot encoding
- **Test**: `KDDTest+.txt` — 22,544 connections (harder distribution, unseen attack types)
- Labels: `normal` → 0, any attack → 1
- Source: [UNB / Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/nsl.html)

### TON_IoT

- **File**: `train_test_network.csv` — 211,044 connections with real IP addresses
- Used to build the trust graph: 776 IP nodes, 211K directed edges
- Source: [UNSW](https://research.unsw.edu.au/projects/toniot-datasets)

### How the datasets work together

```
NSL-KDD (per-connection) ──► Transformer ──► anomaly_score ∈ [0, 1]
TON_IoT (per-entity IP)  ──► GNN         ──► trust_score   ∈ [0, 1]
Both                      ──► Access Decision (Eq. 18) ──► ALLOW / DENY
```

---

## 4. Training

### Default training (NSL-KDD)

```bash
source venv/bin/activate
python3 -m ae_fzta.train
```

This will:
1. Load NSL-KDD (125K connections) + TON_IoT trust graph (776 nodes)
2. Split data across clients using non-IID Dirichlet distribution (α=0.5)
3. Run federated training rounds with FedAvg aggregation
4. Save checkpoints to `./checkpoints/`
5. Auto-generate result plots to `./results/`

### Custom parameters

```python
from ae_fzta.train import run_federated_training

results = run_federated_training(
    dataset="nsl",          # "nsl" or "ton"
    num_clients=20,         # More clients = more privacy, slower convergence
    num_rounds=40,          # More rounds = better convergence
    local_epochs=2,         # Local epochs per client per round
    batch_size=11264,       # Tuned for ~90% GPU on GTX 1650
    checkpoint_every=5,     # Save checkpoint interval
)
```

### How to tell if training is working

| Metric | Bad | Decent | Good |
|---|---|---|---|
| `loss` | Not decreasing / oscillating | Decreasing slowly | Drops to <0.1 |
| `accuracy` | <0.70 or oscillating | 0.80–0.90, stable | >0.90, stable |
| `trust_auc` | <0.60 | 0.70–0.85 | >0.85 |

---

## 5. Testing a Trained Model

```bash
source venv/bin/activate
python3 -m ae_fzta.test_model --checkpoint checkpoints/best_model.npz --dataset nsl
```

This produces:
- Evaluation metrics (accuracy, F1, trust AUC, communication overhead, latency)
- Classification report (per-class precision, recall, F1)
- Confusion matrix heatmap → `./results/test_confusion_matrix.png`
- Sample access decisions with detailed reasoning

### All CLI options

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (required) | Path to `.npz` checkpoint file |
| `--dataset` | `nsl` | `nsl` or `ton` |
| `--anomaly-threshold` | 0.5 | τ_a for access decisions |
| `--trust-threshold` | 0.5 | τ_t for access decisions |
| `--num-samples` | 50 | Number of sample decisions to print |
| `--output-dir` | `./results/` | Output directory for plots |
| `--show-errors-only` | `False` | Only show incorrect predictions |
| `--threshold-sweep` | `False` | Run anomaly threshold sweep (FPR/FNR table) |

---

## 6. LLM Policy Model

The LLM (T5-based) policy generator is trained **separately** from the federated pipeline:

```bash
source venv/bin/activate
python3 -m ae_fzta.train_policy
```

This will:
1. Generate synthetic log-policy pairs from NSL-KDD feature space
2. Fine-tune a T5 model to generate ALLOW/DENY policies from log inputs
3. Save the trained model to `./checkpoints/llm_policy_model/`

The LLM is excluded from federated averaging (too large for efficient parameter exchange). Once trained, it is loaded by `test_model.py` for access decision evaluation.

---

## 7. Visualization

### Auto-generated after training

Training automatically saves plots to `./results/`:
- `training_curves.png` — loss, accuracy, trust AUC per round
- `client_distribution.png` — non-IID data split across clients
- `communication_overhead.png` — per-round and cumulative overhead

### After model testing

`test_model.py` saves:
- `results/test_confusion_matrix.png` — anomaly detection confusion matrix

### Manual generation

```bash
python3 -m ae_fzta.visualize
```

---

## 8. Module Tests

### Full integration test

```bash
python3 ae_fzta/integration_test.py
```

Verifies: imports, config validation, 2-round federated training, evaluation metrics, decision function.

### Individual module tests

```bash
python3 -m ae_fzta.models.transformer_anomaly   # Transformer forward/loss/param round-trip
python3 -m ae_fzta.models.gnn_trust              # GNN forward/loss/param round-trip
python3 -m ae_fzta.decision                      # Decision function test cases
python3 -m ae_fzta.data.preprocessor             # Data loading (NSL + TON + graph)
```

---

## 9. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `FATAL: CUDA is not available` | No GPU or CPU-only PyTorch | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| `ModuleNotFoundError: ae_fzta` | Package not installed | `pip install -e .` from project root |
| `ModuleNotFoundError: matplotlib` | matplotlib not installed | `pip install matplotlib` |
| `CUDA out of memory` | Batch too large for GPU | Reduce `BATCH_SIZE` in `config.py` |
| `FileNotFoundError` on dataset | Wrong working directory | Run from `zai-fed/` project root |
| Accuracy oscillating between rounds | FedAvg instability with non-IID data | Reduce `FL_DIRICHLET_ALPHA`, increase `FL_NUM_ROUNDS`, or increase `FL_FEDPROX_MU` |
| `ae_fzta.egg-info/` appeared | Normal — created by `pip install -e .` | Harmless metadata, add to `.gitignore` |

### Key configuration constants (in `config.py`)

| Constant | Current Value | What It Controls |
|---|---|---|
| `FL_NUM_CLIENTS` | 20 | Number of simulated federated clients |
| `FL_NUM_ROUNDS` | 40 | Number of communication rounds |
| `FL_LOCAL_EPOCHS` | 2 | Local training epochs per client per round |
| `FL_DIRICHLET_ALPHA` | 0.5 | Non-IID skew (lower = more heterogeneous) |
| `BATCH_SIZE` | 11264 | Training batch size (tuned for GTX 1650) |
| `ANOMALY_THRESHOLD` | 0.5 | Attack detection threshold τ_a |
| `TRUST_THRESHOLD` | 0.5 | Trust scoring threshold τ_t |
| `DP_NOISE_SCALE` | 0.001 | Differential privacy Gaussian noise σ |
| `USE_FOCAL_LOSS` | True | Focal loss for class imbalance handling |
