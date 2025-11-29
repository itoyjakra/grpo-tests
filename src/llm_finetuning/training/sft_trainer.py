"""Supervised Fine-Tuning (SFT) trainer module.

This module handles the SFT pre-training phase where the model learns
the custom formatting and basic problem-solving patterns.
"""

import logging
from typing import Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from trl import SFTTrainer, SFTConfig

from ..config import SFTTrainingConfig

logger = logging.getLogger(__name__)


class SFTTrainingPipeline:
    """Manages SFT pre-training pipeline.

    This class handles the supervised fine-tuning phase where the model
    learns to format responses correctly before GRPO training.

    Attributes:
        config: SFT training configuration
    """

    def __init__(self, config: SFTTrainingConfig):
        """Initialize SFT training pipeline.

        Args:
            config: SFT training configuration
        """
        self.config = config
        self.trainer: Optional[SFTTrainer] = None

    def create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> SFTTrainer:
        """Create SFT trainer instance.

        Args:
            model: LoRA-configured model
            tokenizer: Tokenizer with configured chat template
            train_dataset: Training dataset with 'text' field
            eval_dataset: Optional evaluation dataset

        Returns:
            Configured SFTTrainer instance

        Raises:
            ValueError: If dataset doesn't have 'text' field
        """
        if "text" not in train_dataset.column_names:
            raise ValueError("Training dataset must have 'text' field")

        logger.info("Creating SFT trainer...")

        # Create SFT configuration
        sft_config = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_train_epochs,
            learning_rate=self.config.learning_rate,
            logging_steps=self.config.logging_steps,
            optim=self.config.optim,
            weight_decay=self.config.weight_decay,
            lr_scheduler_type=self.config.lr_scheduler_type,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            save_steps=self.config.save_steps,
            report_to=self.config.report_to,
        )

        # Create trainer
        self.trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_config,
        )

        logger.info(
            f"SFT Trainer configured:\n"
            f"  Training examples: {len(train_dataset)}\n"
            f"  Epochs: {self.config.num_train_epochs}\n"
            f"  Batch size: {self.config.per_device_train_batch_size}\n"
            f"  Learning rate: {self.config.learning_rate}\n"
            f"  Output dir: {self.config.output_dir}"
        )

        return self.trainer

    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Execute SFT training.

        Args:
            model: LoRA-configured model
            tokenizer: Tokenizer with configured chat template
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset

        Raises:
            RuntimeError: If training fails
        """
        logger.info("Starting SFT training...")

        try:
            # Create trainer if not already created
            if self.trainer is None:
                self.create_trainer(model, tokenizer, train_dataset, eval_dataset)

            # Run training
            train_result = self.trainer.train()

            logger.info(
                f"SFT training completed:\n"
                f"  Final loss: {train_result.training_loss:.4f}\n"
                f"  Steps: {train_result.global_step}"
            )

        except Exception as e:
            logger.error(f"SFT training failed: {e}")
            raise RuntimeError(f"SFT training failed: {e}")

    def save_model(self, save_path: str) -> None:
        """Save trained model checkpoint.

        Args:
            save_path: Path to save model

        Raises:
            RuntimeError: If no trainer exists or saving fails
        """
        if self.trainer is None:
            raise RuntimeError("No trainer exists. Run training first.")

        logger.info(f"Saving SFT model to {save_path}")

        try:
            self.trainer.save_model(save_path)
            logger.info(f"Model saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Failed to save model: {e}")


def create_sft_trainer(config: SFTTrainingConfig) -> SFTTrainingPipeline:
    """Factory function to create SFT training pipeline.

    Args:
        config: SFT training configuration

    Returns:
        Initialized SFTTrainingPipeline instance
    """
    return SFTTrainingPipeline(config)
