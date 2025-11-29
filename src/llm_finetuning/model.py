"""Model loading and LoRA configuration module.

This module handles loading base models, configuring LoRA adapters,
and saving/loading trained models.
"""

import logging
from typing import Tuple, Optional

import torch
from unsloth import FastLanguageModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import ModelConfig

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading, LoRA configuration, and persistence.

    This class handles all model-related operations including loading
    base models, applying LoRA, and saving/loading trained weights.

    Attributes:
        config: Model configuration
    """

    def __init__(self, config: ModelConfig):
        """Initialize model manager.

        Args:
            config: Model configuration
        """
        self.config = config

    def load_model(self) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load base model and apply LoRA configuration.

        Returns:
            Tuple of (model, tokenizer)

        Raises:
            RuntimeError: If model loading fails
        """
        logger.info(f"Loading model: {self.config.model_name}")

        try:
            # Load base model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_seq_length,
                load_in_4bit=self.config.load_in_4bit,
                fast_inference=self.config.fast_inference,
                max_lora_rank=self.config.lora_rank,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
            )

            logger.info(
                f"Model loaded:\n"
                f"  Model: {self.config.model_name}\n"
                f"  Max sequence length: {self.config.max_seq_length}\n"
                f"  4-bit quantization: {self.config.load_in_4bit}\n"
                f"  GPU memory utilization: {self.config.gpu_memory_utilization}"
            )

            # Apply LoRA
            model = self._apply_lora(model)

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LoRA configuration to model.

        Args:
            model: Base model

        Returns:
            Model with LoRA adapters applied
        """
        logger.info("Applying LoRA configuration...")

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.config.lora_rank,
            target_modules=self.config.target_modules,
            lora_alpha=self.config.lora_alpha,
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            random_state=self.config.random_state,
        )

        logger.info(
            f"LoRA applied:\n"
            f"  Rank: {self.config.lora_rank}\n"
            f"  Alpha: {self.config.lora_alpha}\n"
            f"  Target modules: {', '.join(self.config.target_modules)}\n"
            f"  Gradient checkpointing: {self.config.use_gradient_checkpointing}"
        )

        return model

    def save_lora_weights(
        self,
        model: PreTrainedModel,
        save_path: str
    ) -> None:
        """Save LoRA adapter weights.

        Args:
            model: Model with LoRA adapters
            save_path: Path to save weights

        Raises:
            RuntimeError: If saving fails
        """
        logger.info(f"Saving LoRA weights to {save_path}")

        try:
            model.save_pretrained(save_path)
            logger.info(f"LoRA weights saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save LoRA weights: {e}")
            raise RuntimeError(f"Failed to save LoRA weights: {e}")

    def load_lora_weights(
        self,
        model: PreTrainedModel,
        load_path: str
    ) -> PreTrainedModel:
        """Load LoRA adapter weights.

        Args:
            model: Base model with LoRA configuration
            load_path: Path to load weights from

        Returns:
            Model with loaded LoRA weights

        Raises:
            RuntimeError: If loading fails
        """
        logger.info(f"Loading LoRA weights from {load_path}")

        try:
            model.load_lora(load_path)
            logger.info(f"LoRA weights loaded successfully from {load_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            raise RuntimeError(f"Failed to load LoRA weights: {e}")

    def merge_and_save(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        save_path: str,
        quantization: Optional[str] = None
    ) -> None:
        """Merge LoRA weights with base model and save.

        Args:
            model: Model with LoRA adapters
            tokenizer: Tokenizer to save with model
            save_path: Path to save merged model
            quantization: Optional quantization ('16bit', '4bit', etc.)

        Raises:
            RuntimeError: If merging or saving fails
        """
        logger.info(f"Merging LoRA weights and saving to {save_path}")

        try:
            if quantization == "4bit":
                model.save_pretrained_merged(
                    save_path,
                    tokenizer,
                    save_method="merged_4bit"
                )
            elif quantization == "16bit":
                model.save_pretrained_merged(
                    save_path,
                    tokenizer,
                    save_method="merged_16bit"
                )
            else:
                model.save_pretrained_merged(save_path, tokenizer)

            logger.info(
                f"Merged model saved successfully to {save_path} "
                f"(quantization: {quantization or 'none'})"
            )
        except Exception as e:
            logger.error(f"Failed to merge and save model: {e}")
            raise RuntimeError(f"Failed to merge and save model: {e}")

    def cleanup_memory(self) -> None:
        """Clear GPU memory cache.

        Useful between training phases to free up memory.
        """
        logger.info("Cleaning up GPU memory...")
        torch.cuda.empty_cache()
        logger.info("GPU memory cleaned")


def create_model_manager(config: ModelConfig) -> ModelManager:
    """Factory function to create a model manager.

    Args:
        config: Model configuration

    Returns:
        Initialized ModelManager instance
    """
    return ModelManager(config)
