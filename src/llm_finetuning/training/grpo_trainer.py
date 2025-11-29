"""Group Relative Policy Optimization (GRPO) trainer module.

This module handles the GRPO training phase where the model is trained
using reinforcement learning with reward functions to improve both
formatting and answer correctness.
"""

import logging
from typing import List, Callable, Optional

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from ..config import GRPOTrainingConfig, VLLMSamplingConfig, ModelConfig

logger = logging.getLogger(__name__)


class GRPOTrainingPipeline:
    """Manages GRPO training pipeline.

    This class handles the reinforcement learning phase where the model
    is optimized using reward functions to improve response quality.

    Attributes:
        grpo_config: GRPO training configuration
        vllm_config: vLLM sampling configuration
        model_config: Model configuration
        trainer: GRPO trainer instance
    """

    def __init__(
        self,
        grpo_config: GRPOTrainingConfig,
        vllm_config: VLLMSamplingConfig,
        model_config: ModelConfig
    ):
        """Initialize GRPO training pipeline.

        Args:
            grpo_config: GRPO training configuration
            vllm_config: vLLM sampling configuration
            model_config: Model configuration
        """
        self.grpo_config = grpo_config
        self.vllm_config = vllm_config
        self.model_config = model_config
        self.trainer: Optional[GRPOTrainer] = None

    def create_sampling_params(
        self,
        tokenizer: PreTrainedTokenizer
    ) -> SamplingParams:
        """Create vLLM sampling parameters for generation.

        Args:
            tokenizer: Tokenizer to get EOS token

        Returns:
            Configured SamplingParams instance
        """
        logger.info("Creating vLLM sampling parameters...")

        sampling_params = SamplingParams(
            temperature=self.vllm_config.temperature,
            top_p=self.vllm_config.top_p,
            top_k=self.vllm_config.top_k,
            min_p=self.vllm_config.min_p,
            seed=self.vllm_config.seed,
            stop=[tokenizer.eos_token],
            include_stop_str_in_output=self.vllm_config.include_stop_str_in_output,
        )

        logger.info(
            f"vLLM Sampling configured:\n"
            f"  Temperature: {self.vllm_config.temperature}\n"
            f"  Top-p: {self.vllm_config.top_p}\n"
            f"  Top-k: {self.vllm_config.top_k}\n"
            f"  Min-p: {self.vllm_config.min_p}"
        )

        return sampling_params

    def create_grpo_config(
        self,
        max_prompt_length: int,
        max_completion_length: int,
        sampling_params: SamplingParams
    ) -> GRPOConfig:
        """Create GRPO training configuration.

        Args:
            max_prompt_length: Maximum prompt length in tokens
            max_completion_length: Maximum completion length in tokens
            sampling_params: vLLM sampling parameters

        Returns:
            Configured GRPOConfig instance
        """
        logger.info("Creating GRPO configuration...")

        config = GRPOConfig(
            learning_rate=self.grpo_config.learning_rate,
            weight_decay=self.grpo_config.weight_decay,
            warmup_ratio=self.grpo_config.warmup_ratio,
            lr_scheduler_type=self.grpo_config.lr_scheduler_type,
            optim=self.grpo_config.optim,
            per_device_train_batch_size=self.grpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=self.grpo_config.gradient_accumulation_steps,
            logging_steps=self.grpo_config.logging_steps,
            max_steps=self.grpo_config.max_steps,
            save_steps=self.grpo_config.save_steps,
            output_dir=self.grpo_config.output_dir,
            report_to=self.grpo_config.report_to,
            num_generations=self.grpo_config.num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            vllm_sampling_params=sampling_params,
        )

        # Add optional eval configuration
        if self.grpo_config.eval_strategy != "no":
            config.eval_strategy = self.grpo_config.eval_strategy
            if self.grpo_config.eval_steps is not None:
                config.eval_steps = self.grpo_config.eval_steps

        logger.info(
            f"GRPO Config created:\n"
            f"  Max steps: {self.grpo_config.max_steps}\n"
            f"  Learning rate: {self.grpo_config.learning_rate}\n"
            f"  Num generations: {self.grpo_config.num_generations}\n"
            f"  Max prompt length: {max_prompt_length}\n"
            f"  Max completion length: {max_completion_length}\n"
            f"  Batch size: {self.grpo_config.per_device_train_batch_size}\n"
            f"  Gradient accumulation: {self.grpo_config.gradient_accumulation_steps}"
        )

        return config

    def create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        reward_functions: List[Callable],
        max_prompt_length: int,
        max_completion_length: int,
        eval_dataset: Optional[Dataset] = None
    ) -> GRPOTrainer:
        """Create GRPO trainer instance.

        Args:
            model: LoRA-configured model (should be SFT pre-trained)
            tokenizer: Tokenizer with configured chat template
            train_dataset: Training dataset with 'prompt' and 'answer' fields
            reward_functions: List of reward function callables
            max_prompt_length: Maximum prompt length in tokens
            max_completion_length: Maximum completion length in tokens
            eval_dataset: Optional evaluation dataset

        Returns:
            Configured GRPOTrainer instance

        Raises:
            ValueError: If dataset doesn't have required fields or no reward functions
        """
        required_fields = ["prompt", "answer"]
        for field in required_fields:
            if field not in train_dataset.column_names:
                raise ValueError(f"Training dataset must have '{field}' field")

        if not reward_functions:
            raise ValueError("At least one reward function must be provided")

        logger.info("Creating GRPO trainer...")

        # Create sampling parameters
        sampling_params = self.create_sampling_params(tokenizer)

        # Create GRPO configuration
        grpo_config = self.create_grpo_config(
            max_prompt_length,
            max_completion_length,
            sampling_params
        )

        # Create trainer
        self.trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        logger.info(
            f"GRPO Trainer configured:\n"
            f"  Training examples: {len(train_dataset)}\n"
            f"  Reward functions: {len(reward_functions)}\n"
            f"  Max steps: {self.grpo_config.max_steps}\n"
            f"  Output dir: {self.grpo_config.output_dir}"
        )

        return self.trainer

    def train(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        reward_functions: List[Callable],
        max_prompt_length: int,
        max_completion_length: int,
        eval_dataset: Optional[Dataset] = None
    ) -> None:
        """Execute GRPO training.

        Args:
            model: LoRA-configured model
            tokenizer: Tokenizer with configured chat template
            train_dataset: Training dataset
            reward_functions: List of reward function callables
            max_prompt_length: Maximum prompt length
            max_completion_length: Maximum completion length
            eval_dataset: Optional evaluation dataset

        Raises:
            RuntimeError: If training fails
        """
        logger.info("Starting GRPO training...")
        logger.info(
            "NOTE: Initial rewards will be low as model learns formatting. "
            "Expect improvements after ~150-200 steps."
        )

        try:
            # Create trainer if not already created
            if self.trainer is None:
                self.create_trainer(
                    model,
                    tokenizer,
                    train_dataset,
                    reward_functions,
                    max_prompt_length,
                    max_completion_length,
                    eval_dataset
                )

            # Run training
            train_result = self.trainer.train()

            logger.info(
                f"GRPO training completed:\n"
                f"  Steps: {train_result.global_step}"
            )

        except Exception as e:
            logger.error(f"GRPO training failed: {e}")
            raise RuntimeError(f"GRPO training failed: {e}")

    def save_model(self, save_path: str) -> None:
        """Save trained LoRA weights.

        Args:
            save_path: Path to save LoRA weights

        Raises:
            RuntimeError: If no trainer exists or saving fails
        """
        if self.trainer is None:
            raise RuntimeError("No trainer exists. Run training first.")

        logger.info(f"Saving GRPO model to {save_path}")

        try:
            self.trainer.save_model(save_path)
            logger.info(f"LoRA weights saved successfully to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise RuntimeError(f"Failed to save model: {e}")


def create_grpo_trainer(
    grpo_config: GRPOTrainingConfig,
    vllm_config: VLLMSamplingConfig,
    model_config: ModelConfig
) -> GRPOTrainingPipeline:
    """Factory function to create GRPO training pipeline.

    Args:
        grpo_config: GRPO training configuration
        vllm_config: vLLM sampling configuration
        model_config: Model configuration

    Returns:
        Initialized GRPOTrainingPipeline instance
    """
    return GRPOTrainingPipeline(grpo_config, vllm_config, model_config)
