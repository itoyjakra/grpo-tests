"""Reasoning task evaluation utilities.

This module provides evaluation metrics and utilities for assessing
model performance on reasoning tasks during GRPO training.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for reasoning tasks.

    Attributes:
        format_accuracy: Percentage of outputs with correct format
        answer_accuracy: Percentage of correct answers
        reasoning_present: Percentage with reasoning content
        avg_reasoning_length: Average length of reasoning sections
        avg_total_reward: Average total reward score
        num_samples: Number of samples evaluated
    """
    format_accuracy: float
    answer_accuracy: float
    reasoning_present: float
    avg_reasoning_length: float
    avg_total_reward: float
    num_samples: int

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "eval/format_accuracy": self.format_accuracy,
            "eval/answer_accuracy": self.answer_accuracy,
            "eval/reasoning_present": self.reasoning_present,
            "eval/avg_reasoning_length": self.avg_reasoning_length,
            "eval/avg_total_reward": self.avg_total_reward,
            "eval/num_samples": float(self.num_samples),
        }

    def __str__(self) -> str:
        """Pretty print metrics."""
        return (
            f"Evaluation Metrics (n={self.num_samples}):\n"
            f"  Format Accuracy:    {self.format_accuracy:.1f}%\n"
            f"  Answer Accuracy:    {self.answer_accuracy:.1f}%\n"
            f"  Reasoning Present:  {self.reasoning_present:.1f}%\n"
            f"  Avg Reasoning Len:  {self.avg_reasoning_length:.0f} chars\n"
            f"  Avg Total Reward:   {self.avg_total_reward:.2f}"
        )


class ReasoningEvaluator:
    """Evaluator for reasoning task performance.

    This class evaluates model outputs on reasoning tasks by checking:
    - Format correctness (presence of reasoning and solution tags)
    - Answer correctness (comparing with ground truth)
    - Reasoning quality (presence and length of reasoning)

    Attributes:
        reasoning_start: Start tag for reasoning section
        reasoning_end: End tag for reasoning section
        solution_start: Start tag for solution section
        solution_end: End tag for solution section
    """

    def __init__(
        self,
        reasoning_start: str = "<start_working_out>",
        reasoning_end: str = "<end_working_out>",
        solution_start: str = "<SOLUTION>",
        solution_end: str = "</SOLUTION>"
    ):
        """Initialize evaluator with format tags.

        Args:
            reasoning_start: Start tag for reasoning
            reasoning_end: End tag for reasoning
            solution_start: Start tag for solution
            solution_end: End tag for solution
        """
        self.reasoning_start = reasoning_start
        self.reasoning_end = reasoning_end
        self.solution_start = solution_start
        self.solution_end = solution_end

    def extract_reasoning(self, text: str) -> Optional[str]:
        """Extract reasoning section from output.

        Args:
            text: Model output text

        Returns:
            Reasoning content or None if not found
        """
        pattern = re.escape(self.reasoning_start) + r"(.*?)" + re.escape(self.reasoning_end)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def extract_solution(self, text: str) -> Optional[str]:
        """Extract solution from output.

        Args:
            text: Model output text

        Returns:
            Solution content or None if not found
        """
        pattern = re.escape(self.solution_start) + r"(.*?)" + re.escape(self.solution_end)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def check_format(self, text: str) -> bool:
        """Check if output has correct format.

        Args:
            text: Model output text

        Returns:
            True if format is correct
        """
        has_reasoning = (
            self.reasoning_start in text and
            self.reasoning_end in text
        )
        has_solution = (
            self.solution_start in text and
            self.solution_end in text
        )
        return has_reasoning and has_solution

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison.

        Strips whitespace, converts to lowercase, removes punctuation.

        Args:
            answer: Answer string

        Returns:
            Normalized answer
        """
        # Remove common punctuation and whitespace
        answer = answer.strip().lower()
        answer = re.sub(r'[.,!?;:]', '', answer)
        answer = re.sub(r'\s+', ' ', answer)
        return answer

    def check_answer(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer

        Returns:
            True if answers match
        """
        pred_norm = self.normalize_answer(predicted)
        truth_norm = self.normalize_answer(ground_truth)

        # Exact match after normalization
        if pred_norm == truth_norm:
            return True

        # Check if ground truth is contained in prediction
        if truth_norm in pred_norm:
            return True

        # Check for numerical equivalence
        try:
            pred_num = float(re.sub(r'[^\d.-]', '', predicted))
            truth_num = float(re.sub(r'[^\d.-]', '', ground_truth))
            return abs(pred_num - truth_num) < 0.001
        except (ValueError, TypeError):
            pass

        return False

    def evaluate_single(
        self,
        output: str,
        ground_truth: str
    ) -> Tuple[bool, bool, bool, int]:
        """Evaluate a single model output.

        Args:
            output: Model output text
            ground_truth: Ground truth answer

        Returns:
            Tuple of (format_correct, answer_correct, has_reasoning, reasoning_length)
        """
        # Check format
        format_correct = self.check_format(output)

        # Extract reasoning
        reasoning = self.extract_reasoning(output)
        has_reasoning = reasoning is not None and len(reasoning) > 0
        reasoning_length = len(reasoning) if reasoning else 0

        # Extract and check answer
        solution = self.extract_solution(output)
        answer_correct = False
        if solution:
            answer_correct = self.check_answer(solution, ground_truth)

        return format_correct, answer_correct, has_reasoning, reasoning_length

    def evaluate_batch(
        self,
        outputs: List[str],
        ground_truths: List[str],
        rewards: Optional[List[float]] = None
    ) -> EvaluationMetrics:
        """Evaluate a batch of model outputs.

        Args:
            outputs: List of model outputs
            ground_truths: List of ground truth answers
            rewards: Optional list of reward scores

        Returns:
            EvaluationMetrics with aggregated results
        """
        if len(outputs) != len(ground_truths):
            raise ValueError("Outputs and ground truths must have same length")

        format_correct_count = 0
        answer_correct_count = 0
        reasoning_present_count = 0
        total_reasoning_length = 0

        for output, truth in zip(outputs, ground_truths):
            format_ok, answer_ok, has_reasoning, reasoning_len = self.evaluate_single(
                output, truth
            )

            if format_ok:
                format_correct_count += 1
            if answer_ok:
                answer_correct_count += 1
            if has_reasoning:
                reasoning_present_count += 1
            total_reasoning_length += reasoning_len

        n = len(outputs)
        avg_reward = sum(rewards) / n if rewards else 0.0

        return EvaluationMetrics(
            format_accuracy=100.0 * format_correct_count / n,
            answer_accuracy=100.0 * answer_correct_count / n,
            reasoning_present=100.0 * reasoning_present_count / n,
            avg_reasoning_length=total_reasoning_length / n,
            avg_total_reward=avg_reward,
            num_samples=n
        )


def create_evaluator(
    reasoning_start: str = "<start_working_out>",
    reasoning_end: str = "<end_working_out>",
    solution_start: str = "<SOLUTION>",
    solution_end: str = "</SOLUTION>"
) -> ReasoningEvaluator:
    """Factory function to create reasoning evaluator.

    Args:
        reasoning_start: Start tag for reasoning
        reasoning_end: End tag for reasoning
        solution_start: Start tag for solution
        solution_end: End tag for solution

    Returns:
        Configured ReasoningEvaluator instance
    """
    return ReasoningEvaluator(
        reasoning_start=reasoning_start,
        reasoning_end=reasoning_end,
        solution_start=solution_start,
        solution_end=solution_end
    )
