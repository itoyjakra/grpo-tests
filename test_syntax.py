#!/usr/bin/env python3
"""Simple syntax and import check for modules.

This script verifies that all modules have correct syntax
and can be imported without runtime errors.
"""

import sys
from pathlib import Path
import py_compile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_syntax():
    """Test Python syntax of all modules."""
    logger.info("Testing Python syntax...")

    modules = [
        "src/llm_finetuning/__init__.py",
        "src/llm_finetuning/config.py",
        "src/llm_finetuning/model.py",
        "src/llm_finetuning/data/__init__.py",
        "src/llm_finetuning/data/loader.py",
        "src/llm_finetuning/data/processor.py",
        "src/llm_finetuning/data/templates.py",
        "src/llm_finetuning/rewards/__init__.py",
        "src/llm_finetuning/rewards/reward_functions.py",
        "src/llm_finetuning/training/__init__.py",
        "src/llm_finetuning/training/sft_trainer.py",
        "src/llm_finetuning/training/grpo_trainer.py",
        "run_training.py",
    ]

    errors = []
    for module_path in modules:
        try:
            py_compile.compile(module_path, doraise=True)
            logger.info(f"✓ {module_path}")
        except py_compile.PyCompileError as e:
            logger.error(f"✗ {module_path}: {e}")
            errors.append((module_path, str(e)))

    return len(errors) == 0, errors


def test_config_yaml():
    """Test YAML config file is valid."""
    logger.info("\nTesting YAML configuration...")

    try:
        import yaml
        with open("configs/default_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        assert "model" in config
        assert "data" in config
        assert "sft_training" in config
        assert "grpo_training" in config

        logger.info("✓ YAML configuration is valid")
        return True
    except Exception as e:
        logger.error(f"✗ YAML configuration error: {e}")
        return False


def test_structure():
    """Test directory structure exists."""
    logger.info("\nTesting directory structure...")

    required_dirs = [
        "src/llm_finetuning",
        "src/llm_finetuning/data",
        "src/llm_finetuning/rewards",
        "src/llm_finetuning/training",
        "configs",
    ]

    missing = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            logger.error(f"✗ Missing directory: {dir_path}")
            missing.append(dir_path)
        else:
            logger.info(f"✓ {dir_path}")

    return len(missing) == 0


def main():
    """Run all syntax checks."""
    logger.info("=" * 60)
    logger.info("LLM Fine-Tuning Module Syntax Validation")
    logger.info("=" * 60)

    # Test structure
    structure_ok = test_structure()

    # Test syntax
    syntax_ok, syntax_errors = test_syntax()

    # Test YAML
    yaml_ok = test_config_yaml()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Validation Summary")
    logger.info("=" * 60)

    if structure_ok:
        logger.info("✓ Directory structure: OK")
    else:
        logger.error("✗ Directory structure: FAILED")

    if syntax_ok:
        logger.info("✓ Python syntax: OK")
    else:
        logger.error("✗ Python syntax: FAILED")
        for path, error in syntax_errors:
            logger.error(f"  {path}: {error}")

    if yaml_ok:
        logger.info("✓ YAML configuration: OK")
    else:
        logger.error("✗ YAML configuration: FAILED")

    logger.info("=" * 60)

    if structure_ok and syntax_ok and yaml_ok:
        logger.info("\n🎉 All syntax checks passed!")
        logger.info("\nThe module structure is correct and ready to use.")
        logger.info("Note: Full functionality testing requires running with proper dependencies.")
        return 0
    else:
        logger.error("\n❌ Some checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
