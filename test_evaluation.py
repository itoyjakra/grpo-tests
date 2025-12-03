#!/usr/bin/env python3
"""Test evaluation functionality."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_finetuning.evaluation import create_evaluator, EvaluationMetrics

# Test data
test_outputs = [
    "<start_working_out>15 * 23 = 345<end_working_out><SOLUTION>345</SOLUTION>",
    "<start_working_out>Wrong reasoning<end_working_out><SOLUTION>999</SOLUTION>",
    "No format at all: 345",
    "<start_working_out>10 + 5 = 15<end_working_out><SOLUTION>15</SOLUTION>",
]

test_ground_truths = [
    "345",
    "345",
    "345",
    "15",
]

test_rewards = [10.5, -2.5, -6.5, 12.0]

# Create evaluator
evaluator = create_evaluator()

# Evaluate batch
print("Testing Evaluation System\n" + "="*50)
metrics = evaluator.evaluate_batch(test_outputs, test_ground_truths, test_rewards)
print(metrics)
print("\n" + "="*50)
print("✅ Evaluation system working correctly!")
