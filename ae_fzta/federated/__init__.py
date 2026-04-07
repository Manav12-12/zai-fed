"""AE-FZTA Federated Package.

Exports:
    ZTAFederatedClient: Flower NumPyClient.
    encrypt_weights / decrypt_weights: AES-256-GCM weight encryption.
    fedavg_aggregate: FedAvg parameter aggregation.
    FederatedCoordinator: FL orchestration helper.
"""

from ae_fzta.federated.client import (
    ZTAFederatedClient,
    encrypt_weights,
    decrypt_weights,
)
from ae_fzta.federated.server import fedavg_aggregate, FederatedCoordinator

__all__ = [
    "ZTAFederatedClient",
    "encrypt_weights",
    "decrypt_weights",
    "fedavg_aggregate",
    "FederatedCoordinator",
]
