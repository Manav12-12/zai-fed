"""Synthetic Policy Data Generator.

Generates synthetic log-policy training pairs derived from the NSL-KDD
feature space for contextually consistent LLM training.
"""

import logging
import random
from datetime import datetime, timedelta
from typing import List, Tuple

logger = logging.getLogger(__name__)

# NSL-KDD protocol, service, and flag values as specified
PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES_STANDARD = ["http", "ftp", "ssh", "smtp"]
SERVICES_SUSPICIOUS = ["domain_u", "private"]
FLAGS_ALLOW = ["SF"]
FLAGS_DENY = ["S0", "REJ"]


def _generate_synthetic_ip() -> str:
    """Generate a random synthetic IPv4 address."""
    return f"{random.randint(10, 192)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"


def _generate_allow_pair() -> Tuple[str, str]:
    """Generate a safe log and its corresponding ALLOW policy."""
    timestamp_dt = datetime.now() - timedelta(days=random.randint(0, 30))
    # Business hours timestamps
    hour = random.randint(9, 17)
    timestamp = timestamp_dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)).isoformat() + "Z"
    
    protocol = random.choice(PROTOCOLS)
    # Ensure standard service and flag
    service = random.choice(SERVICES_STANDARD)
    flag = random.choice(FLAGS_ALLOW)
    
    # Moderate bytes
    src_bytes = random.randint(100, 15000)
    dst_bytes = random.randint(100, 50000)
    duration = random.randint(0, 120)

    ip = _generate_synthetic_ip()

    log_str = (
        f"timestamp={timestamp} protocol_type={protocol} service={service} "
        f"flag={flag} src_bytes={src_bytes} dst_bytes={dst_bytes} duration={duration}"
    )
    
    # Exact format: ACTION user:IDENTIFIER resource:SERVICE condition:REASON
    policy_str = f"ALLOW user:{ip} resource:{service} condition:Standard_business_access"
    return log_str, policy_str


def _generate_deny_pair() -> Tuple[str, str]:
    """Generate a suspicious log and its corresponding DENY policy."""
    timestamp_dt = datetime.now() - timedelta(days=random.randint(0, 30))
    protocol = random.choice(PROTOCOLS)
    service = random.choice(SERVICES_STANDARD)
    flag = random.choice(FLAGS_ALLOW)
    src_bytes = random.randint(100, 15000)
    dst_bytes = random.randint(100, 50000)
    duration = random.randint(0, 120)
    
    hour = random.randint(9, 17)
    
    # Pick AT LEAST ONE realistic threat indicator
    threat_type = random.randint(0, 4)
    reason = "Anomalous_activity"

    if threat_type == 0:
        # Off-hours timestamp between 01:00 and 05:00
        hour = random.randint(1, 5)
        reason = "Off_hours_access"
    elif threat_type == 1:
        # Source byte count exceeding 500000
        src_bytes = random.randint(500001, 10000000)
        reason = "High_source_bytes"
    elif threat_type == 2:
        # Flag value of S0 or REJ
        flag = random.choice(FLAGS_DENY)
        reason = "Suspicious_connection_flag"
    elif threat_type == 3:
        # Service of domain_u or private
        service = random.choice(SERVICES_SUSPICIOUS)
        reason = "Suspicious_service_port"
    elif threat_type == 4:
        # ICMP protocol with large byte counts
        protocol = "icmp"
        src_bytes = random.randint(50000, 500000)
        reason = "Large_icmp_payload"
        
    timestamp = timestamp_dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)).isoformat() + "Z"
    ip = _generate_synthetic_ip()
    
    log_str = (
        f"timestamp={timestamp} protocol_type={protocol} service={service} "
        f"flag={flag} src_bytes={src_bytes} dst_bytes={dst_bytes} duration={duration}"
    )
    
    policy_str = f"DENY user:{ip} resource:{service} condition:{reason}"
    return log_str, policy_str


def generate_synthetic_policies(
    num_pairs: int,
    allow_ratio: float,
    seed: int
) -> Tuple[List[str], List[str]]:
    """Generate paired synthetic logs and policies mapped to NSL-KDD.

    Args:
        num_pairs: Total count of log-policy pairs to generate.
        allow_ratio: Target proportion of ALLOW policies.
        seed: Random seed.

    Returns:
        logs (List[str]), policies (List[str])
    """
    random.seed(seed)
    
    num_allow = int(num_pairs * allow_ratio)
    num_deny = num_pairs - num_allow
    
    logs: List[str] = []
    policies: List[str] = []

    for _ in range(num_allow):
        l, p = _generate_allow_pair()
        logs.append(l)
        policies.append(p)
        
    for _ in range(num_deny):
        l, p = _generate_deny_pair()
        logs.append(l)
        policies.append(p)
        
    combined = list(zip(logs, policies))
    random.shuffle(combined)
    
    logs, policies = zip(*combined)
    logs = list(logs)
    policies = list(policies)
    
    actual_allow_ratio = num_allow / num_pairs
    if num_pairs >= 20:
        assert abs(actual_allow_ratio - allow_ratio) <= 0.05, "ALLOW ratio deviated by > 5%"
    
    logger.info(
        "Generated %d synthetic pairs. ALLOW: %.2f%%, DENY: %.2f%%",
        num_pairs,
        actual_allow_ratio * 100.0,
        (1 - actual_allow_ratio) * 100.0,
    )
    
    return logs, policies


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    test_logs, test_policies = generate_synthetic_policies(
        num_pairs=50,
        allow_ratio=0.5,
        seed=123
    )
    
    assert len(test_logs) == len(test_policies) == 50, "Lengths mismatch or incorrect."
    assert all(len(p) > 0 for p in test_policies), "Found an empty policy string."
    
    print("Example pairs:")
    for i in range(5):
        print(f"LOG   : {test_logs[i]}")
        print(f"POLICY: {test_policies[i]}\n")
