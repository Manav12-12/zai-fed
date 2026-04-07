"""AE-FZTA Models Package.

Exports:
    TransformerAnomalyDetector: TBAE model.
    GNNTrustModel: GNTE model.
    LLMPolicyGenerator: PALLM model (preserved, not used in training).
"""

from ae_fzta.models.transformer_anomaly import TransformerAnomalyDetector
from ae_fzta.models.gnn_trust import GNNTrustModel

# LLM preserved but not imported by default to avoid heavy deps
# from ae_fzta.models.llm_policy import LLMPolicyGenerator

__all__ = [
    "TransformerAnomalyDetector",
    "GNNTrustModel",
]
