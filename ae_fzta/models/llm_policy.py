"""LLM-based Policy Automation Module (PALLM).

Implements the LLM-based policy generation described in Section IV-D
of the AE-FZTA paper (Paper Equations 13–14). Uses a T5 model to
automatically generate Zero Trust access control policies from
unstructured system logs.

Classes:
    LLMPolicyGenerator: T5-based policy generation and fine-tuning.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

logger = logging.getLogger(__name__)


class LLMPolicyGenerator:
    """T5-based Zero Trust policy generator.

    Implements Paper Equations 13–14 for automated policy synthesis
    from unstructured log data.

    Eq. 13: π_i = f_θ(l_i)  — policy generation from log input.
    Eq. 14: L_LLM = -Σ log P_θ(π_i | l_i) — negative log-likelihood loss.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        device: Optional[torch.device] = None,
        max_length: int = 128,
        beam_width: int = 4,
    ) -> None:
        """Initialise the policy generator with a pre-trained T5 model.

        Loads both model and tokenizer and moves the model to the
        specified device. Device handling is encapsulated within this class.

        Args:
            model_name: HuggingFace T5 model identifier.
            device: Torch device to use. Defaults to CPU if not specified.
            max_length: Maximum token length for generation.
            beam_width: Number of beams for beam search decoding.
        """
        self.device = device if device is not None else torch.device("cpu")
        self.max_length = max_length
        self.beam_width = beam_width
        self.model_name = model_name

        import os
        from pathlib import Path

        if os.path.isdir(model_name) or os.path.exists(model_name) or getattr(model_name, "startswith", lambda x: False)("./") or "/" in str(model_name):
            abs_path = os.path.abspath(model_name)
            if not Path(abs_path).exists():
                raise FileNotFoundError(f"LLM Policy Checkpoint not found at: {abs_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(abs_path, local_files_only=True)
            self.model = T5ForConditionalGeneration.from_pretrained(abs_path, local_files_only=True)
        else:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
        self.model.to(self.device)

        logger.info(
            "LLMPolicyGenerator: model=%s, device=%s, max_length=%d, beams=%d",
            model_name,
            self.device,
            max_length,
            beam_width,
        )

    def fine_tune(
        self,
        log_texts: List[str],
        policy_texts: List[str],
        epochs: int = 3,
        learning_rate: float = 5e-5,
        batch_size: int = 8,
    ) -> List[float]:
        """Fine-tune the T5 model on log-to-policy pairs (Paper Eq. 14).

        The T5 model natively returns a loss when labels are provided,
        corresponding to the negative log-likelihood formulation of
        Eq. 14: L_LLM = -Σ log P_θ(π_i | l_i).

        Args:
            log_texts: List of input log strings.
            policy_texts: List of target policy strings.
            epochs: Number of training epochs.
            learning_rate: Optimiser learning rate.
            batch_size: Training batch size.

        Returns:
            List of average loss values per epoch.
        """
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        epoch_losses: List[float] = []
        num_samples = len(log_texts)

        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_logs = log_texts[start:end]
                batch_policies = policy_texts[start:end]

                # Tokenise inputs
                inputs = self.tokenizer(
                    batch_logs,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.device)

                # Tokenise targets
                targets = self.tokenizer(
                    batch_policies,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                # Mask padding in labels with -100
                label_ids = targets["input_ids"].clone()
                label_ids[label_ids == self.tokenizer.pad_token_id] = -100
                label_ids = label_ids.to(self.device)

                # Forward pass — T5 computes CE loss natively (Eq. 14)
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=label_ids,
                )
                loss = outputs.loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            logger.info("LLM fine-tune epoch %d/%d — avg_loss=%.4f", epoch + 1, epochs, avg_loss)

        self.model.eval()
        return epoch_losses

    def generate_policy(self, log_text: str) -> str:
        """Generate a Zero Trust policy from a log string (Paper Eq. 13).

        Uses beam search decoding with special tokens stripped.

        Args:
            log_text: Raw log text input.

        Returns:
            Generated policy string with special tokens stripped.
        """
        self.model.eval()
        inputs = self.tokenizer(
            log_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.max_length,
                num_beams=self.beam_width,
                early_stopping=True,
            )

        # Decode and strip special tokens
        policy = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return policy

    def get_numpy_parameters(self) -> List[np.ndarray]:
        """Extract model parameters as a list of numpy arrays.

        Returns:
            List of numpy arrays, one per parameter tensor, preserving
            parameter ordering for exact restoration.
        """
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_numpy_parameters(self, params: List[np.ndarray]) -> None:
        """Restore model parameters from numpy arrays.

        Args:
            params: List of numpy arrays matching model parameter shapes
                and ordering from get_numpy_parameters().
        """
        for param, np_arr in zip(self.model.parameters(), params):
            param.data = torch.tensor(np_arr, dtype=param.dtype, device=param.device)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Only test inference, not fine-tuning, to keep runtime reasonable
    gen = LLMPolicyGenerator(
        model_name="t5-small",
        device=torch.device("cpu"),
        max_length=64,
        beam_width=2,
    )

    test_log = (
        "generate policy: 2025-01-15T10:30:00Z user=admin action=READ "
        "resource=database ip=192.168.1.100"
    )
    policy = gen.generate_policy(test_log)

    # Assert output is a non-empty string
    assert isinstance(policy, str), f"Expected str, got {type(policy)}"
    assert len(policy) > 0, "Generated policy is empty"

    # Test parameter round-trip
    params = gen.get_numpy_parameters()
    assert len(params) > 0, "No parameters extracted"
    gen.set_numpy_parameters(params)
    policy_after = gen.generate_policy(test_log)
    assert isinstance(policy_after, str) and len(policy_after) > 0

    logger.info("Generated policy: %s", policy)
    logger.info("Num LLM parameters: %d arrays", len(params))
    logger.info("✅ STEP 7 COMPLETE — llm_policy.py all tests passed.")
