"""Dataset processing and formatting for LLM fine-tuning.

This module handles formatting datasets into chat message format,
tokenization, filtering by length, and conversion to HuggingFace format.
"""

import logging
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizer

from ..config import DataConfig, ModelConfig

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """Processes and formats datasets for training.

    This class handles:
    - Formatting examples into chat message format
    - Tokenization and length-based filtering
    - Conversion to HuggingFace Dataset format

    Attributes:
        data_config: Data configuration
        model_config: Model configuration (for max_seq_length)
    """

    def __init__(self, data_config: DataConfig, model_config: ModelConfig):
        """Initialize dataset processor.

        Args:
            data_config: Data configuration
            model_config: Model configuration
        """
        self.data_config = data_config
        self.model_config = model_config

    def format_sft_example(
        self,
        row: pd.Series,
        system_prompt: str,
        prompt_components: Dict[str, str]
    ) -> List[Dict[str, str]]:
        """Format a single example into chat message format for SFT.

        Converts a dataset row into a list of messages with system, user,
        and assistant roles. Replaces generic <think> tags with custom
        formatting tags.

        Args:
            row: Pandas Series with 'expected_answer', 'problem', 'generated_solution'
            system_prompt: System prompt to use
            prompt_components: Dictionary of formatting tags

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        expected_answer = row["expected_answer"]
        problem = row["problem"]
        thoughts = row["generated_solution"]

        # Remove generic <think> and </think> tags
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")
        thoughts = thoughts.strip()

        # Extract formatting tags
        reasoning_start = prompt_components["reasoning_start"]
        reasoning_end = prompt_components["reasoning_end"]
        solution_start = prompt_components["solution_start"]
        solution_end = prompt_components["solution_end"]

        # Create formatted response
        final_prompt = (
            f"{reasoning_start}{thoughts}{reasoning_end}"
            f"{solution_start}{expected_answer}{solution_end}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": final_prompt},
        ]

    def process_sft_dataset(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str,
        prompt_components: Dict[str, str]
    ) -> Dataset:
        """Process complete SFT dataset.

        Applies formatting, tokenization, length filtering, and conversion
        to HuggingFace Dataset format.

        Args:
            df: Pandas DataFrame with SFT examples
            tokenizer: Tokenizer for length calculation
            system_prompt: System prompt to use
            prompt_components: Dictionary of formatting tags

        Returns:
            HuggingFace Dataset ready for SFT training

        Raises:
            ValueError: If no examples remain after filtering
        """
        logger.info("Processing SFT dataset...")

        # Format all examples
        df["Messages"] = df.apply(
            lambda x: self.format_sft_example(x, system_prompt, prompt_components),
            axis=1
        )

        # Calculate lengths and filter
        df = self._truncate_by_length(df, tokenizer)

        # Convert to HuggingFace format
        dataset = self._make_hf_compatible(df, tokenizer)

        logger.info(f"Final SFT dataset size: {len(dataset)} examples")
        return dataset

    def process_grpo_dataset(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizer
    ) -> tuple[Dataset, int, int]:
        """Process GRPO dataset and calculate sequence length constraints.

        Tokenizes prompts, calculates max_prompt_length based on percentile,
        and filters dataset to keep only suitable examples.

        Args:
            dataset: HuggingFace Dataset with 'prompt' field
            tokenizer: Tokenizer for length calculation

        Returns:
            Tuple of (filtered_dataset, max_prompt_length, max_completion_length)

        Raises:
            ValueError: If dataset has no 'prompt' field or no examples remain
        """
        logger.info("Processing GRPO dataset...")

        if "prompt" not in dataset.column_names:
            raise ValueError("GRPO dataset must have 'prompt' field")

        # Tokenize all prompts to calculate lengths
        logger.info("Tokenizing prompts to calculate lengths...")

        def tokenize_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
            """Tokenize a single prompt."""
            tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": example["prompt"]}],
                tokenize=True,
                add_generation_prompt=True
            )
            example["prompt_length"] = len(tokens)
            return example

        dataset = dataset.map(tokenize_prompt, desc="Tokenizing prompts")

        # Calculate percentile-based max_prompt_length
        prompt_lengths = dataset["prompt_length"]
        max_prompt_length = int(
            np.percentile(prompt_lengths, self.data_config.max_length_percentile)
        )

        logger.info(
            f"Prompt length statistics:\n"
            f"  Min: {min(prompt_lengths)}\n"
            f"  Max: {max(prompt_lengths)}\n"
            f"  Mean: {np.mean(prompt_lengths):.1f}\n"
            f"  {self.data_config.max_length_percentile}th percentile: {max_prompt_length}"
        )

        # Filter dataset to keep only suitable examples
        original_size = len(dataset)
        dataset = dataset.filter(
            lambda x: x["prompt_length"] <= max_prompt_length,
            desc="Filtering by prompt length"
        )
        filtered_size = len(dataset)

        logger.info(
            f"Filtered {original_size - filtered_size} long prompts "
            f"({filtered_size}/{original_size} remaining)"
        )

        if filtered_size == 0:
            raise ValueError("No examples remain after length filtering")

        # Calculate max_completion_length
        max_completion_length = self.model_config.max_seq_length - max_prompt_length

        logger.info(
            f"Sequence length allocation:\n"
            f"  max_prompt_length: {max_prompt_length}\n"
            f"  max_completion_length: {max_completion_length}\n"
            f"  total: {self.model_config.max_seq_length}"
        )

        # Rename 'solution' to 'answer' for GRPO trainer compatibility
        if "solution" in dataset.column_names and "answer" not in dataset.column_names:
            dataset = dataset.rename_column("solution", "answer")
            logger.info("Renamed 'solution' column to 'answer' for GRPO compatibility")

        return dataset, max_prompt_length, max_completion_length

    def _truncate_by_length(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer
    ) -> pd.DataFrame:
        """Filter dataset by sequence length.

        Args:
            df: DataFrame with 'Messages' column
            tokenizer: Tokenizer for length calculation

        Returns:
            Filtered DataFrame
        """
        # Calculate token lengths
        df["N"] = df["Messages"].apply(
            lambda x: len(tokenizer.apply_chat_template(x))
        )

        # Filter by truncate_ratio * max_seq_length
        max_length = int(
            self.model_config.max_seq_length * self.data_config.truncate_ratio
        )
        original_size = len(df)
        df = df.loc[df["N"] <= max_length].copy()
        filtered_size = len(df)

        logger.info(
            f"Truncated dataset: max_length={max_length}, "
            f"kept {filtered_size}/{original_size} examples"
        )

        if filtered_size == 0:
            raise ValueError("No examples remain after truncation")

        return df

    def _make_hf_compatible(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer
    ) -> Dataset:
        """Convert DataFrame to HuggingFace Dataset format.

        Args:
            df: DataFrame with 'Messages' column
            tokenizer: Tokenizer to apply chat template

        Returns:
            HuggingFace Dataset with 'text' field
        """
        # Apply chat template to create text field
        df["text"] = tokenizer.apply_chat_template(
            df["Messages"].values.tolist(),
            tokenize=False
        )

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)

        logger.info(f"Created HuggingFace dataset with {len(dataset)} examples")
        return dataset


def create_dataset_processor(
    data_config: DataConfig,
    model_config: ModelConfig
) -> DatasetProcessor:
    """Factory function to create a dataset processor.

    Args:
        data_config: Data configuration
        model_config: Model configuration

    Returns:
        Initialized DatasetProcessor instance
    """
    return DatasetProcessor(data_config, model_config)
