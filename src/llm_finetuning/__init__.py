"""LLM Fine-Tuning Module.

A comprehensive framework for fine-tuning large language models using
Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).
"""

__version__ = "0.1.0"

from .config import (
    ModelConfig,
    DataConfig,
    PromptConfig,
    SFTTrainingConfig,
    GRPOTrainingConfig,
    VLLMSamplingConfig,
    RewardConfig,
    EvaluationConfig,
    PipelineConfig,
)
from .model import ModelManager, create_model_manager

__all__ = [
    "ModelConfig",
    "DataConfig",
    "PromptConfig",
    "SFTTrainingConfig",
    "GRPOTrainingConfig",
    "VLLMSamplingConfig",
    "RewardConfig",
    "EvaluationConfig",
    "PipelineConfig",
    "ModelManager",
    "create_model_manager",
]
