#!/usr/bin/env python3
"""Main training pipeline script.

This script orchestrates the complete fine-tuning pipeline:
1. Load model and configure LoRA
2. SFT pre-training phase
3. GRPO training phase
4. Model saving and evaluation
"""

import argparse
import logging
import sys
from pathlib import Path

import gc
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_finetuning.config import PipelineConfig
from llm_finetuning.model import create_model_manager
from llm_finetuning.data import (
    create_dataset_loader,
    create_dataset_processor,
    create_prompt_manager,
)
from llm_finetuning.rewards import create_reward_manager
from llm_finetuning.training import create_sft_trainer, create_grpo_trainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


class LLMFineTuningPipeline:
    """Complete end-to-end fine-tuning pipeline.

    This class orchestrates the entire training process from model loading
    through SFT and GRPO training to final model saving.

    Attributes:
        config: Complete pipeline configuration
    """

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def run(self) -> None:
        """Execute complete training pipeline.

        Raises:
            RuntimeError: If any phase fails
        """
        logger.info("=" * 80)
        logger.info("Starting LLM Fine-Tuning Pipeline")
        logger.info("=" * 80)

        try:
            # Phase 0: Load model
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 0: Model Loading")
            logger.info("=" * 80)
            self._load_model()

            # Phase 1: SFT Pre-training
            if not self.config.skip_sft:
                logger.info("\n" + "=" * 80)
                logger.info("PHASE 1: SFT Pre-Training")
                logger.info("=" * 80)
                self._run_sft_training()
                self._cleanup_memory()
            else:
                logger.info("\n" + "=" * 80)
                logger.info("PHASE 1: SFT Pre-Training [SKIPPED]")
                logger.info("=" * 80)

            # Phase 2: GRPO Training
            if not self.config.skip_grpo:
                logger.info("\n" + "=" * 80)
                logger.info("PHASE 2: GRPO Training")
                logger.info("=" * 80)
                self._run_grpo_training()
            else:
                logger.info("\n" + "=" * 80)
                logger.info("PHASE 2: GRPO Training [SKIPPED]")
                logger.info("=" * 80)

            # Phase 3: Save final model
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 3: Saving Final Model")
            logger.info("=" * 80)
            self._save_final_model()

            logger.info("\n" + "=" * 80)
            logger.info("Pipeline completed successfully!")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Pipeline failed: {e}")

    def _load_model(self) -> None:
        """Load model and apply LoRA configuration."""
        model_manager = create_model_manager(self.config.model)
        self.model, self.tokenizer = model_manager.load_model()

        # Configure prompt template
        prompt_manager = create_prompt_manager(self.config.prompt)
        chat_template = prompt_manager.configure_chat_template(self.tokenizer)

        logger.info(f"Chat template configured:\n{chat_template[:200]}...")

    def _run_sft_training(self) -> None:
        """Execute SFT pre-training phase."""
        # Create managers
        data_loader = create_dataset_loader(self.config.data)
        data_processor = create_dataset_processor(
            self.config.data,
            self.config.model
        )
        prompt_manager = create_prompt_manager(self.config.prompt)

        # Load and process SFT dataset
        logger.info("Loading SFT dataset...")
        sft_df = data_loader.load_sft_dataset()

        logger.info("Processing SFT dataset...")
        sft_dataset = data_processor.process_sft_dataset(
            sft_df,
            self.tokenizer,
            prompt_manager.get_system_prompt(),
            prompt_manager.get_components()
        )

        # Create and run SFT trainer
        sft_trainer = create_sft_trainer(self.config.sft_training)
        sft_trainer.train(self.model, self.tokenizer, sft_dataset)

        # Save SFT checkpoint
        sft_save_path = f"{self.config.sft_training.output_dir}/final"
        sft_trainer.save_model(sft_save_path)
        logger.info(f"SFT checkpoint saved to {sft_save_path}")

    def _run_grpo_training(self) -> None:
        """Execute GRPO training phase."""
        # Create managers
        data_loader = create_dataset_loader(self.config.data)
        data_processor = create_dataset_processor(
            self.config.data,
            self.config.model
        )
        prompt_manager = create_prompt_manager(self.config.prompt)

        # Load and process GRPO dataset
        logger.info("Loading GRPO dataset...")
        grpo_dataset = data_loader.load_grpo_dataset()

        logger.info("Processing GRPO dataset...")
        grpo_dataset, max_prompt_length, max_completion_length = (
            data_processor.process_grpo_dataset(grpo_dataset, self.tokenizer)
        )

        # Create reward functions
        reward_manager = create_reward_manager(
            prompt_manager.get_components(),
            self.config.reward.print_every_steps
        )
        reward_functions = reward_manager.get_reward_functions(
            use_format_exact=self.config.reward.use_format_exact,
            use_format_approximate=self.config.reward.use_format_approximate,
            use_answer_check=self.config.reward.use_answer_check,
            use_number_check=self.config.reward.use_number_check,
        )

        # Create and run GRPO trainer
        grpo_trainer = create_grpo_trainer(
            self.config.grpo_training,
            self.config.vllm_sampling,
            self.config.model
        )
        grpo_trainer.train(
            self.model,
            self.tokenizer,
            grpo_dataset,
            reward_functions,
            max_prompt_length,
            max_completion_length
        )

        # Save GRPO checkpoint
        grpo_save_path = f"{self.config.grpo_training.output_dir}/final"
        grpo_trainer.save_model(grpo_save_path)
        logger.info(f"GRPO checkpoint saved to {grpo_save_path}")

    def _save_final_model(self) -> None:
        """Save final trained model."""
        model_manager = create_model_manager(self.config.model)
        model_manager.save_lora_weights(self.model, self.config.save_path)
        logger.info(f"Final model saved to {self.config.save_path}")

    def _cleanup_memory(self) -> None:
        """Clean up GPU memory between phases."""
        logger.info("Cleaning up GPU memory...")
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Memory cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fine-tune LLM using SFT and GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_training.py

  # Run with custom config
  python run_training.py --config configs/my_config.yaml

  # Skip SFT (load pretrained SFT model)
  python run_training.py --skip-sft

  # Skip GRPO (only run SFT)
  python run_training.py --skip-grpo

  # Override specific parameters
  python run_training.py --learning-rate 1e-4 --max-steps 200
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        help="Skip SFT pre-training phase"
    )
    parser.add_argument(
        "--skip-grpo",
        action="store_true",
        help="Skip GRPO training phase"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        help="Path to save final model (overrides config)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="GRPO learning rate (overrides config)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="GRPO max steps (overrides config)"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        help="Number of generations per prompt (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = PipelineConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except FileNotFoundError:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = PipelineConfig()

    # Apply CLI overrides
    if args.skip_sft:
        config.skip_sft = True
    if args.skip_grpo:
        config.skip_grpo = True
    if args.save_path:
        config.save_path = args.save_path
    if args.learning_rate:
        config.grpo_training.learning_rate = args.learning_rate
    if args.max_steps:
        config.grpo_training.max_steps = args.max_steps
    if args.num_generations:
        config.grpo_training.num_generations = args.num_generations

    # Log configuration
    logger.info("Pipeline Configuration:")
    logger.info(f"  Model: {config.model.model_name}")
    logger.info(f"  Skip SFT: {config.skip_sft}")
    logger.info(f"  Skip GRPO: {config.skip_grpo}")
    logger.info(f"  Save path: {config.save_path}")
    logger.info(f"  SFT epochs: {config.sft_training.num_train_epochs}")
    logger.info(f"  GRPO max steps: {config.grpo_training.max_steps}")
    logger.info(f"  GRPO learning rate: {config.grpo_training.learning_rate}")

    # Run pipeline
    pipeline = LLMFineTuningPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
