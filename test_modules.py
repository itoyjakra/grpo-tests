#!/usr/bin/env python3
"""Test script to verify module implementation.

This script tests that all modules can be imported and basic
functionality works without running full training.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing module imports...")

    try:
        # Test config imports
        from llm_finetuning import (
            ModelConfig,
            DataConfig,
            PromptConfig,
            SFTTrainingConfig,
            GRPOTrainingConfig,
            VLLMSamplingConfig,
            RewardConfig,
            EvaluationConfig,
            PipelineConfig,
            ModelManager,
            create_model_manager,
        )

        # Test data module imports
        from llm_finetuning.data import (
            DatasetLoader,
            create_dataset_loader,
            DatasetProcessor,
            create_dataset_processor,
            PromptTemplateManager,
            create_prompt_manager,
        )

        # Test rewards module imports
        from llm_finetuning.rewards import (
            RewardFunctionManager,
            create_reward_manager,
        )

        # Test training module imports
        from llm_finetuning.training import (
            SFTTrainingPipeline,
            create_sft_trainer,
            GRPOTrainingPipeline,
            create_grpo_trainer,
        )

        logger.info("✓ All module imports successful")
        return True

    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_config():
    """Test configuration system."""
    logger.info("\nTesting configuration system...")

    try:
        from llm_finetuning import PipelineConfig

        # Test default config creation
        config = PipelineConfig()
        assert config.model.model_name == "unsloth/Qwen3-4B-Base"
        assert config.model.lora_rank == 32
        assert config.sft_training.learning_rate == 2e-4
        assert config.grpo_training.learning_rate == 5e-6

        logger.info("✓ Default configuration works")

        # Test YAML loading
        config_from_yaml = PipelineConfig.from_yaml("configs/default_config.yaml")
        assert config_from_yaml.model.model_name == "unsloth/Qwen3-4B-Base"

        logger.info("✓ YAML configuration loading works")
        return True

    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        return False


def test_prompt_templates():
    """Test prompt template system."""
    logger.info("\nTesting prompt templates...")

    try:
        from llm_finetuning import PromptConfig
        from llm_finetuning.data import create_prompt_manager

        config = PromptConfig()
        prompt_manager = create_prompt_manager(config)

        # Test system prompt creation
        system_prompt = prompt_manager.get_system_prompt()
        assert "<start_working_out>" in system_prompt
        assert "<SOLUTION>" in system_prompt

        # Test components
        components = prompt_manager.get_components()
        assert "reasoning_start" in components
        assert components["reasoning_start"] == "<start_working_out>"

        logger.info("✓ Prompt template system works")
        return True

    except Exception as e:
        logger.error(f"✗ Prompt template test failed: {e}")
        return False


def test_reward_functions():
    """Test reward functions."""
    logger.info("\nTesting reward functions...")

    try:
        from llm_finetuning import PromptConfig
        from llm_finetuning.data import create_prompt_manager
        from llm_finetuning.rewards import create_reward_manager

        prompt_config = PromptConfig()
        prompt_manager = create_prompt_manager(prompt_config)
        components = prompt_manager.get_components()

        reward_manager = create_reward_manager(components)

        # Test exact format matching
        completions_good = [
            "<start_working_out>test<end_working_out><SOLUTION>42</SOLUTION>"
        ]
        completions_bad = ["no formatting here"]

        rewards_good = reward_manager.match_format_exactly(completions_good)
        rewards_bad = reward_manager.match_format_exactly(completions_bad)

        assert rewards_good[0] == 3.0
        assert rewards_bad[0] == 0.0

        logger.info("✓ Reward functions work")
        return True

    except Exception as e:
        logger.error(f"✗ Reward function test failed: {e}")
        return False


def test_factory_functions():
    """Test factory functions."""
    logger.info("\nTesting factory functions...")

    try:
        from llm_finetuning import (
            ModelConfig,
            DataConfig,
            PromptConfig,
            SFTTrainingConfig,
            GRPOTrainingConfig,
            VLLMSamplingConfig,
        )
        from llm_finetuning.model import create_model_manager
        from llm_finetuning.data import (
            create_dataset_loader,
            create_dataset_processor,
            create_prompt_manager,
        )
        from llm_finetuning.rewards import create_reward_manager
        from llm_finetuning.training import (
            create_sft_trainer,
            create_grpo_trainer,
        )

        # Test all factory functions
        model_manager = create_model_manager(ModelConfig())
        dataset_loader = create_dataset_loader(DataConfig())
        dataset_processor = create_dataset_processor(DataConfig(), ModelConfig())
        prompt_manager = create_prompt_manager(PromptConfig())
        reward_manager = create_reward_manager(prompt_manager.get_components())
        sft_trainer = create_sft_trainer(SFTTrainingConfig())
        grpo_trainer = create_grpo_trainer(
            GRPOTrainingConfig(),
            VLLMSamplingConfig(),
            ModelConfig()
        )

        logger.info("✓ All factory functions work")
        return True

    except Exception as e:
        logger.error(f"✗ Factory function test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("Testing LLM Fine-Tuning Module Implementation")
    logger.info("=" * 60)

    tests = [
        ("Module Imports", test_imports),
        ("Configuration System", test_config),
        ("Prompt Templates", test_prompt_templates),
        ("Reward Functions", test_reward_functions),
        ("Factory Functions", test_factory_functions),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)

    if passed == total:
        logger.info("\n🎉 All tests passed! Module implementation is working.")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
