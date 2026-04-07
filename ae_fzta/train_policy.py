"""Standalone training pipeline for the LLM policy model.

Runs independently of the federated loop and produces a saved model
that the federated pipeline can load.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset, DataLoader

from ae_fzta import config
from ae_fzta.models.llm_policy import LLMPolicyGenerator
from ae_fzta.data.policy_generator import generate_synthetic_policies

logger = logging.getLogger(__name__)


class PolicyDataset(Dataset):
    """Dataset for validating generated log-policy pairs."""
    def __init__(self, logs: List[str], policies: List[str], tokenizer, max_length: int = 128):
        self.logs = logs
        self.policies = policies
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.logs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            self.logs[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = self.tokenizer(
            self.policies[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_ids = targets["input_ids"].clone()
        label_ids[label_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label_ids.squeeze(0),
        }


def train_policy_model() -> Dict[str, Any]:
    """Train the LLM policy model.
    
    Performs data generation, dataset splitting, training via fine_tune(),
    validation loop, and model checkpoint saving.
    """
    # Generate synthetic pairs using config constants and fixed seed
    logs, policies = generate_synthetic_policies(
        num_pairs=config.POLICY_SYNTHETIC_PAIRS,
        allow_ratio=config.POLICY_ALLOW_RATIO,
        seed=config.RANDOM_SEED,
    )

    # Split into train and validation
    split_idx = int(len(logs) * config.POLICY_TRAIN_SPLIT)
    train_logs, val_logs = logs[:split_idx], logs[split_idx:]
    train_policies, val_policies = policies[:split_idx], policies[split_idx:]

    logger.info("Splitting dataset: %d train, %d val", len(train_logs), len(val_logs))

    # Instantiate LLMPolicyGenerator using config constants
    generator = LLMPolicyGenerator(
        model_name=config.LLM_MODEL_NAME,
        device=config.DEVICE,
        max_length=config.LLM_MAX_LENGTH,
        beam_width=config.LLM_BEAM_WIDTH,
    )

    # Create PolicyDataset for both splits using the tokenizer
    train_dataset = PolicyDataset(train_logs, train_policies, generator.tokenizer, config.LLM_MAX_LENGTH)
    val_dataset = PolicyDataset(val_logs, val_policies, generator.tokenizer, config.LLM_MAX_LENGTH)

    # Create DataLoaders with config batch size and shuffle enabled only for training
    train_loader = DataLoader(train_dataset, batch_size=config.LLM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.LLM_BATCH_SIZE, shuffle=False)

    epoch_train_losses = []
    epoch_val_losses = []

    # Outer loop for epochs to interleve validation 
    for epoch in range(config.LLM_EPOCHS):
        logger.info("\n--- Epoch %d/%d ---", epoch + 1, config.LLM_EPOCHS)
        
        # Call the existing fine_tune() method on the training DataLoader (using the raw lists to satisfy the method signature)
        # Note: We pass raw lists because fine_tune() intrinsically requires List[str].
        train_loss_list = generator.fine_tune(
            log_texts=train_logs,
            policy_texts=train_policies,
            epochs=1,
            learning_rate=config.LLM_LEARNING_RATE,
            batch_size=config.LLM_BATCH_SIZE,
        )
        avg_train_loss = train_loss_list[0]
        epoch_train_losses.append(avg_train_loss)

        # Evaluate validation loss by running the model in no-gradient mode on the validation DataLoader
        generator.model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(config.DEVICE)
                attention_mask = batch["attention_mask"].to(config.DEVICE)
                labels = batch["labels"].to(config.DEVICE)

                outputs = generator.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_val_loss += outputs.loss.item()
                val_batches += 1
                
        avg_val_loss = total_val_loss / max(val_batches, 1)
        epoch_val_losses.append(avg_val_loss)

        # Log both training and validation loss after every epoch
        logger.info(
            "Epoch %d summary: Train Loss = %.4f | Val Loss = %.4f", 
            epoch + 1, avg_train_loss, avg_val_loss
        )

    # Save the model and the tokenizer to the configured save directory creating it if it does not exist
    save_dir = Path(os.path.abspath(config.POLICY_MODEL_SAVE_PATH))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    generator.model.save_pretrained(save_dir, safe_serialization=False)
    generator.tokenizer.save_pretrained(save_dir, safe_serialization=False)
    logger.info("Saved trained LLM model and tokenizer to %s", save_dir)
    
    # Return a dictionary
    return {
        "train_losses": epoch_train_losses,
        "val_losses": epoch_val_losses,
        "saved_path": str(save_dir.resolve()),
    }


def load_trained_policy_model(save_dir: str) -> LLMPolicyGenerator:
    """Load a ready-to-use LLMPolicyGenerator instance from disk."""
    abs_path = os.path.abspath(save_dir)
    if not Path(abs_path).exists():
        raise FileNotFoundError(f"LLM Policy Checkpoint not found at: {abs_path}")
        
    try:
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        # The tokenizer and model both need to explicitly block hub lookups
        model = T5ForConditionalGeneration.from_pretrained(abs_path, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(abs_path, local_files_only=True)
        
        generator = LLMPolicyGenerator(
            model_name="t5-small", # initialize empty
            device=config.DEVICE,
            max_length=config.LLM_MAX_LENGTH,
            beam_width=config.LLM_BEAM_WIDTH,
        )
        generator.model = model.to(config.DEVICE)
        generator.tokenizer = tokenizer
        return generator
    except (OSError, EnvironmentError) as e:
        logger.error(f"HuggingFace failed to load the LLM directory: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    # Call train_policy_model, print the returned dictionary
    results_dict = train_policy_model()
    print("\nTraining Results Dictionary:")
    print(results_dict)
    
    # Immediately call load_trained_policy_model pointing to the saved path
    loaded_generator = load_trained_policy_model(results_dict["saved_path"])
    
    # Call generate_policy() on one synthetic log string
    sample_log = "timestamp=2026-02-17T09:16:51Z protocol_type=tcp service=http flag=SF src_bytes=10491 dst_bytes=2764 duration=0"
    generated_pol = loaded_generator.generate_policy(sample_log)
    
    # Print the output
    print(f"\nGenerated Policy for test string: {generated_pol}")
    
    # Assert it is a non-empty string containing either ALLOW or DENY
    assert isinstance(generated_pol, str) and len(generated_pol) > 0, "Generated policy must be a non-empty string"
    assert "ALLOW" in generated_pol.upper() or "DENY" in generated_pol.upper(), "Policy must contain ALLOW or DENY"
