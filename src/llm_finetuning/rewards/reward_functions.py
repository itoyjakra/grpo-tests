"""Reward functions for GRPO training.

This module implements reward functions that evaluate model completions
based on formatting correctness and answer accuracy.
"""

import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class RewardFunctionManager:
    """Manages reward functions for GRPO training.

    This class encapsulates all reward functions and provides a unified
    interface for creating reward function lists based on configuration.

    Attributes:
        prompt_components: Dictionary of formatting tags
        print_every_steps: Print debug info every N steps (0 = never)
        printed_times: Counter for debug printing
    """

    def __init__(
        self,
        prompt_components: Dict[str, str],
        print_every_steps: int = 0
    ):
        """Initialize reward function manager.

        Args:
            prompt_components: Dictionary with formatting tags
            print_every_steps: Print debug info every N steps (0 = never)
        """
        self.prompt_components = prompt_components
        self.print_every_steps = print_every_steps
        self.printed_times = 0

        # Compile regex patterns
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for answer extraction."""
        # Pattern for exact format matching
        self.exact_format_pattern = re.compile(
            rf"{re.escape(self.prompt_components['reasoning_start'])}"
            r".*?"
            rf"{re.escape(self.prompt_components['reasoning_end'])}"
            rf"{re.escape(self.prompt_components['solution_start'])}"
            r".*?"
            rf"{re.escape(self.prompt_components['solution_end'])}",
            re.DOTALL
        )

        # Pattern for extracting solution
        self.solution_pattern = re.compile(
            rf"{re.escape(self.prompt_components['solution_start'])}"
            r"(.*?)"
            rf"{re.escape(self.prompt_components['solution_end'])}",
            re.DOTALL
        )

        # Pattern for extracting numbers from solution
        self.number_pattern = re.compile(r"-?\d+\.?\d*")

    def match_format_exactly(
        self,
        completions: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[float]:
        """Check if completions match the exact expected format.

        Awards +3.0 points if the completion matches the exact pattern:
        <reasoning_start>...<reasoning_end><solution_start>...<solution_end>

        Args:
            completions: List of completion message lists (each is [{"role": "assistant", "content": "..."}])
            **kwargs: Additional arguments (unused)

        Returns:
            List of rewards (+3.0 for exact match, 0.0 otherwise)
        """
        rewards = []
        for completion in completions:
            # Extract content from message dict
            response = completion[0]["content"]
            if self.exact_format_pattern.search(response):
                rewards.append(3.0)
            else:
                rewards.append(0.0)
        return rewards

    def match_format_approximately(
        self,
        completions: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[float]:
        """Score completions based on presence of formatting tags.

        Awards points based on tag counts:
        - +0.5 for each correct tag occurrence (max +1.5 for 3 tags)
        - -1.0 penalty for missing or duplicate tags

        Expected: exactly 1 occurrence each of reasoning_end, solution_start, solution_end

        Args:
            completions: List of completion message lists
            **kwargs: Additional arguments (unused)

        Returns:
            List of rewards based on tag presence
        """
        rewards = []
        for completion in completions:
            reward = 0.0

            # Extract content from message dict
            response = completion[0]["content"]

            # Count occurrences of each tag (excluding reasoning_start which is in prompt)
            reasoning_end_count = response.count(self.prompt_components["reasoning_end"])
            solution_start_count = response.count(self.prompt_components["solution_start"])
            solution_end_count = response.count(self.prompt_components["solution_end"])

            # Award points for correct single occurrences
            if reasoning_end_count == 1:
                reward += 0.5
            else:
                reward -= 1.0

            if solution_start_count == 1:
                reward += 0.5
            else:
                reward -= 1.0

            if solution_end_count == 1:
                reward += 0.5
            else:
                reward -= 1.0

            rewards.append(reward)
        return rewards

    def check_answer(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: List[str],
        **kwargs
    ) -> List[float]:
        """Extract and score answer correctness.

        Scoring rubric:
        - +5.0: Exact string match
        - +3.5: Match with whitespace differences
        - +2.0: Numerical close match (ratio 0.9-1.1)
        - +1.5: Numerical reasonably close (ratio 0.8-1.2)
        - -2.5: Wrong answer
        - -2.0: Cannot extract answer

        Args:
            prompts: List of prompt message lists
            completions: List of completion message lists
            answer: List of expected answers
            **kwargs: Additional arguments (unused)

        Returns:
            List of rewards based on answer correctness
        """
        rewards = []

        for i, (completion, expected) in enumerate(zip(completions, answer)):
            # Extract content from message dict
            response = completion[0]["content"]

            # Try to extract solution from completion
            match = self.solution_pattern.search(response)

            if not match:
                rewards.append(-2.0)
                self._debug_print(i, response, None, expected, "NO_MATCH")
                continue

            extracted = match.group(1).strip()

            # Check for exact match
            if extracted == expected:
                rewards.append(5.0)
                self._debug_print(i, response, extracted, expected, "EXACT")
                continue

            # Check for match with whitespace differences
            if extracted.replace(" ", "") == expected.replace(" ", ""):
                rewards.append(3.5)
                self._debug_print(i, response, extracted, expected, "WHITESPACE")
                continue

            # Try numerical comparison
            try:
                extracted_num = float(extracted)
                expected_num = float(expected)

                if expected_num == 0:
                    # Avoid division by zero
                    if extracted_num == 0:
                        rewards.append(5.0)
                        self._debug_print(i, response, extracted, expected, "EXACT_ZERO")
                    else:
                        rewards.append(-2.5)
                        self._debug_print(i, response, extracted, expected, "WRONG")
                    continue

                ratio = extracted_num / expected_num

                if 0.9 <= ratio <= 1.1:
                    rewards.append(2.0)
                    self._debug_print(i, response, extracted, expected, "CLOSE")
                elif 0.8 <= ratio <= 1.2:
                    rewards.append(1.5)
                    self._debug_print(i, response, extracted, expected, "REASONABLE")
                else:
                    rewards.append(-2.5)
                    self._debug_print(i, response, extracted, expected, "WRONG")

            except (ValueError, TypeError):
                # Cannot convert to number
                rewards.append(-2.5)
                self._debug_print(i, response, extracted, expected, "NON_NUMERIC")

        return rewards

    def check_numbers(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: List[str],
        **kwargs
    ) -> List[float]:
        """Alternative numerical answer checking function.

        Similar to check_answer but with different scoring weights and
        focuses on numerical extraction.

        Scoring rubric:
        - +4.0: Exact numerical match
        - +2.5: Close match (ratio 0.9-1.1)
        - +1.5: Reasonable match (ratio 0.8-1.2)
        - -2.0: Wrong answer
        - -1.5: Cannot extract number

        Args:
            prompts: List of prompt message lists
            completions: List of completion message lists
            answer: List of expected answers
            **kwargs: Additional arguments (unused)

        Returns:
            List of rewards based on numerical correctness
        """
        rewards = []

        for i, (completion, expected) in enumerate(zip(completions, answer)):
            # Extract content from message dict
            response = completion[0]["content"]

            # Try to extract solution
            match = self.solution_pattern.search(response)

            if not match:
                rewards.append(-1.5)
                continue

            extracted = match.group(1).strip()

            # Try to extract numbers from the solution
            extracted_numbers = self.number_pattern.findall(extracted)

            if not extracted_numbers:
                rewards.append(-1.5)
                continue

            # Use the first number found
            try:
                extracted_num = float(extracted_numbers[0])
                expected_num = float(expected)

                if expected_num == 0:
                    if extracted_num == 0:
                        rewards.append(4.0)
                    else:
                        rewards.append(-2.0)
                    continue

                ratio = extracted_num / expected_num

                if abs(extracted_num - expected_num) < 1e-6:
                    rewards.append(4.0)
                elif 0.9 <= ratio <= 1.1:
                    rewards.append(2.5)
                elif 0.8 <= ratio <= 1.2:
                    rewards.append(1.5)
                else:
                    rewards.append(-2.0)

            except (ValueError, TypeError):
                rewards.append(-1.5)

        return rewards

    def _debug_print(
        self,
        index: int,
        completion: str,
        extracted: Optional[str],
        expected: str,
        result_type: str
    ) -> None:
        """Print debug information periodically.

        Args:
            index: Example index
            completion: Full completion
            extracted: Extracted answer
            expected: Expected answer
            result_type: Type of result (EXACT, CLOSE, WRONG, etc.)
        """
        if self.print_every_steps > 0:
            if self.printed_times % self.print_every_steps == 0:
                logger.info(
                    f"[{result_type}] Example {index}:\n"
                    f"  Extracted: {extracted}\n"
                    f"  Expected: {expected}\n"
                    f"  Completion: {completion[:200]}..."
                )
            self.printed_times += 1

    def get_reward_functions(
        self,
        use_format_exact: bool = True,
        use_format_approximate: bool = True,
        use_answer_check: bool = True,
        use_number_check: bool = True
    ) -> List[callable]:
        """Get list of reward functions based on configuration.

        Args:
            use_format_exact: Include exact format matching
            use_format_approximate: Include approximate format matching
            use_answer_check: Include answer correctness checking
            use_number_check: Include numerical answer checking

        Returns:
            List of reward function callables
        """
        reward_funcs = []

        if use_format_exact:
            reward_funcs.append(self.match_format_exactly)

        if use_format_approximate:
            reward_funcs.append(self.match_format_approximately)

        if use_answer_check:
            reward_funcs.append(self.check_answer)

        if use_number_check:
            reward_funcs.append(self.check_numbers)

        logger.info(f"Configured {len(reward_funcs)} reward functions")
        return reward_funcs


def create_reward_manager(
    prompt_components: Dict[str, str],
    print_every_steps: int = 0
) -> RewardFunctionManager:
    """Factory function to create a reward function manager.

    Args:
        prompt_components: Dictionary of formatting tags
        print_every_steps: Print debug info every N steps

    Returns:
        Initialized RewardFunctionManager instance
    """
    return RewardFunctionManager(prompt_components, print_every_steps)
