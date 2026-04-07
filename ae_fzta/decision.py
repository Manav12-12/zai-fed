"""Access Control Decision Module.

Implements the final access decision function from Paper Eq. 18.
Combines anomaly score, trust score, and LLM-generated policy to make
a binary allow/deny decision for each access request.

d_i = 𝟙[y'_i < τ_a ∧ T(v_i) > τ_t ∧ π_i(r_i) = ALLOW]

Functions:
    make_access_decision: Single-request access decision.
    make_batch_decisions: Batch access decisions.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Token that must appear in the policy string (case-insensitive)
_ALLOW_TOKEN = "allow"


def _validate_score(name: str, value: float) -> None:
    """Validate that a score is in [0, 1].

    Args:
        name: Name of the score for error messages.
        value: Score value to validate.

    Raises:
        ValueError: If the score is outside [0, 1].
    """
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def make_access_decision(
    anomaly_score: float,
    trust_score: float,
    policy_string: str,
    anomaly_threshold: float,
    trust_threshold: float,
) -> bool:
    """Make an access control decision per Paper Eq. 18.

    Access is granted only if ALL three conditions are met:
    1. anomaly_score < anomaly_threshold (strictly less than)
    2. trust_score > trust_threshold (strictly greater than)
    3. policy_string contains 'ALLOW' (case-insensitive)

    Args:
        anomaly_score: Predicted anomaly score in [0, 1].
        trust_score: Predicted trust score in [0, 1].
        policy_string: Generated policy text.
        anomaly_threshold: Maximum acceptable anomaly score (τ_a in Eq. 18).
        trust_threshold: Minimum acceptable trust score (τ_t in Eq. 18).

    Returns:
        True if access is granted, False otherwise.

    Raises:
        ValueError: If scores are outside [0, 1].
    """
    _validate_score("anomaly_score", anomaly_score)
    _validate_score("trust_score", trust_score)

    # Evaluate each condition individually (Paper Eq. 18)
    cond_anomaly = anomaly_score < anomaly_threshold
    cond_trust = trust_score > trust_threshold
    cond_policy = _ALLOW_TOKEN in policy_string.lower()

    logger.info(
        "Decision: anomaly=%.4f<%s=%.4f → %s, trust=%.4f>%s=%.4f → %s, "
        "policy_contains_ALLOW → %s",
        anomaly_score,
        "τ_a",
        anomaly_threshold,
        cond_anomaly,
        trust_score,
        "τ_t",
        trust_threshold,
        cond_trust,
        cond_policy,
    )

    decision = cond_anomaly and cond_trust and cond_policy
    return decision


def make_batch_decisions(
    anomaly_scores: List[float],
    trust_scores: List[float],
    policy_strings: List[str],
    anomaly_threshold: float,
    trust_threshold: float,
) -> List[bool]:
    """Make batch access decisions per Paper Eq. 18.

    Args:
        anomaly_scores: List of anomaly scores in [0, 1].
        trust_scores: List of trust scores in [0, 1].
        policy_strings: List of policy strings.
        anomaly_threshold: Maximum acceptable anomaly score (τ_a).
        trust_threshold: Minimum acceptable trust score (τ_t).

    Returns:
        List of boolean decisions.

    Raises:
        ValueError: If input lists have unequal lengths.
    """
    if not (len(anomaly_scores) == len(trust_scores) == len(policy_strings)):
        raise ValueError(
            f"Input lists must have equal length: anomaly={len(anomaly_scores)}, "
            f"trust={len(trust_scores)}, policy={len(policy_strings)}"
        )

    return [
        make_access_decision(a, t, p, anomaly_threshold, trust_threshold)
        for a, t, p in zip(anomaly_scores, trust_scores, policy_strings)
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Test 1: All conditions pass ---
    result = make_access_decision(
        anomaly_score=0.2,
        trust_score=0.8,
        policy_string="POLICY: decision=ALLOW",
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert result is True, f"Expected True, got {result}"

    # --- Test 2: Anomaly condition fails ---
    result = make_access_decision(
        anomaly_score=0.7,
        trust_score=0.8,
        policy_string="POLICY: decision=ALLOW",
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert result is False, f"Expected False (anomaly too high), got {result}"

    # --- Test 3: Trust condition fails ---
    result = make_access_decision(
        anomaly_score=0.2,
        trust_score=0.3,
        policy_string="POLICY: decision=ALLOW",
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert result is False, f"Expected False (trust too low), got {result}"

    # --- Test 4: Policy condition fails ---
    result = make_access_decision(
        anomaly_score=0.2,
        trust_score=0.8,
        policy_string="POLICY: decision=DENY",
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert result is False, f"Expected False (no ALLOW), got {result}"

    # --- Test 5: Edge case — score equals threshold exactly ---
    # anomaly_score == threshold → NOT strictly less than → DENY
    result = make_access_decision(
        anomaly_score=0.5,
        trust_score=0.8,
        policy_string="POLICY: decision=ALLOW",
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert result is False, f"Expected False (anomaly == threshold), got {result}"

    # trust_score == threshold → NOT strictly greater than → DENY
    result = make_access_decision(
        anomaly_score=0.2,
        trust_score=0.5,
        policy_string="POLICY: decision=ALLOW",
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert result is False, f"Expected False (trust == threshold), got {result}"

    # --- Test 6: Invalid scores raise ValueError ---
    try:
        make_access_decision(1.5, 0.5, "ALLOW", 0.5, 0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        make_access_decision(0.5, -0.1, "ALLOW", 0.5, 0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # --- Test 7: Batch decisions ---
    results = make_batch_decisions(
        anomaly_scores=[0.2, 0.8, 0.2],
        trust_scores=[0.8, 0.8, 0.3],
        policy_strings=["ALLOW", "ALLOW", "ALLOW"],
        anomaly_threshold=0.5,
        trust_threshold=0.5,
    )
    assert results == [True, False, False], f"Expected [True, False, False], got {results}"

    # --- Test 8: Batch length mismatch ---
    try:
        make_batch_decisions([0.2], [0.8, 0.3], ["ALLOW"], 0.5, 0.5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # --- Test 9: Case-insensitive ALLOW ---
    result = make_access_decision(0.2, 0.8, "policy: allow access", 0.5, 0.5)
    assert result is True, f"Expected True (case-insensitive allow), got {result}"

    logger.info("✅ STEP 10 COMPLETE — decision.py all tests passed.")
