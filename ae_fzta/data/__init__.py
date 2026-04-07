"""AE-FZTA Data Package.

Exports:
    load_nsl_kdd, load_ton_iot, build_trust_graph: Dataset loaders.
    split_non_iid, split_iid: Federated data splitting.
    generate_synthetic_data: Synthetic fallback.
    ConnectionDataset: Per-connection PyTorch Dataset.
    GraphDataset: Trust graph PyTorch Dataset.
"""

from ae_fzta.data.preprocessor import (
    load_nsl_kdd,
    load_ton_iot,
    build_trust_graph,
    split_non_iid,
    split_iid,
    generate_synthetic_data,
)
from ae_fzta.data.dataset import ConnectionDataset, GraphDataset

__all__ = [
    "load_nsl_kdd",
    "load_ton_iot",
    "build_trust_graph",
    "split_non_iid",
    "split_iid",
    "generate_synthetic_data",
    "ConnectionDataset",
    "GraphDataset",
]
