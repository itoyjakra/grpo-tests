"""Dataset loading utilities for LLM fine-tuning.

This module handles loading datasets for SFT and GRPO training phases,
including filtering and preprocessing steps.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset

from ..config import DataConfig

logger = logging.getLogger(__name__)


class DatasetLoader:
    """Handles loading and initial preprocessing of datasets.

    This class manages loading datasets from HuggingFace and applying
    initial filters (e.g., numeric-only answers for SFT).

    Attributes:
        config: Data configuration specifying datasets and filters
    """

    def __init__(self, config: DataConfig):
        """Initialize dataset loader.

        Args:
            config: Data configuration
        """
        self.config = config

    def load_sft_dataset(self) -> pd.DataFrame:
        """Load and filter dataset for SFT pre-training.

        Loads the OpenMathReasoning dataset and optionally filters to
        only include examples with numeric answers.

        Returns:
            Pandas DataFrame with columns: expected_answer, problem, generated_solution

        Raises:
            ValueError: If dataset loading fails or no examples remain after filtering
        """
        logger.info(f"Loading SFT dataset: {self.config.sft_dataset_name}")

        try:
            dataset = load_dataset(
                self.config.sft_dataset_name,
                split=self.config.sft_dataset_split
            )
        except Exception as e:
            raise ValueError(f"Failed to load SFT dataset: {e}")

        # Convert to pandas for easier manipulation
        df = dataset.to_pandas()[["expected_answer", "problem", "generated_solution"]]
        logger.info(f"Loaded {len(df)} examples from SFT dataset")

        # Filter to only numeric answers if configured
        if self.config.filter_numeric_only:
            df = self._filter_numeric_answers(df)
            logger.info(f"After numeric filtering: {len(df)} examples")

        if len(df) == 0:
            raise ValueError("No examples remain after filtering")

        return df

    def load_grpo_dataset(self) -> Dataset:
        """Load dataset for GRPO training.

        Loads the DAPO-Math dataset for reinforcement learning training.

        Returns:
            HuggingFace Dataset with columns: prompt, solution, answer

        Raises:
            ValueError: If dataset loading fails
        """
        logger.info(f"Loading GRPO dataset: {self.config.grpo_dataset_name}")

        try:
            dataset = load_dataset(
                self.config.grpo_dataset_name,
                self.config.grpo_dataset_config,
                split=self.config.grpo_dataset_split
            )
        except Exception as e:
            raise ValueError(f"Failed to load GRPO dataset: {e}")

        logger.info(f"Loaded {len(dataset)} examples from GRPO dataset")
        return dataset

    def _filter_numeric_answers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataset to only include numeric answers.

        Args:
            df: DataFrame with 'expected_answer' column

        Returns:
            Filtered DataFrame containing only rows with numeric answers
        """
        # Try converting to number - if not, replace with NaN
        is_number = pd.to_numeric(
            pd.Series(df["expected_answer"]),
            errors="coerce"
        ).notnull()

        # Select only numeric answers
        filtered_df = df.iloc[np.where(is_number)[0]].copy()

        logger.info(
            f"Filtered {len(df) - len(filtered_df)} non-numeric answers "
            f"({len(filtered_df)}/{len(df)} remaining)"
        )

        return filtered_df


def create_dataset_loader(config: DataConfig) -> DatasetLoader:
    """Factory function to create a dataset loader.

    Args:
        config: Data configuration

    Returns:
        Initialized DatasetLoader instance
    """
    return DatasetLoader(config)
