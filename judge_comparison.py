"""
AI Judge for Base vs SFT Model Comparison using AWS Bedrock.

This script:
1. Loads comparison results from JSON file
2. Loads ground truth solutions from exercises.jsonl
3. Uses AWS Bedrock model to judge which model performed better
4. Generates a comprehensive report with statistics

Usage:
    python judge_comparison.py comparison_results_TIMESTAMP.json

Requirements:
    - AWS credentials configured (via ~/.aws/credentials or environment variables)
    - boto3 installed: pip install boto3
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path
import boto3
from typing import Dict, List, Tuple


class BedrockJudge:
    """AWS Bedrock judge for model comparison."""

    def __init__(self, model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0", region: str = "us-east-1"):
        """
        Initialize Bedrock judge.

        Args:
            model_id: Bedrock model ID to use as judge
            region: AWS region
        """
        self.model_id = model_id
        self.client = boto3.client('bedrock-runtime', region_name=region)

    def judge_responses(
        self,
        problem: str,
        ground_truth: str,
        base_response: str,
        sft_response: str
    ) -> Dict:
        """
        Judge which model response is better.

        Args:
            problem: The optimization problem
            ground_truth: The correct solution
            base_response: Base model's response
            sft_response: Fine-tuned model's response

        Returns:
            Dictionary with judgment results
        """
        judge_prompt = f"""You are an expert judge evaluating two AI models' solutions to a convex optimization problem.

**Problem:**
{problem}

**Ground Truth Solution:**
{ground_truth}

**Model A Response (Base Model):**
{base_response}

**Model B Response (Fine-tuned Model):**
{sft_response}

**Evaluation Criteria:**
1. **Format Adherence** (30%): Does the response follow the required format with reasoning tags?
   - Required: `<start_working_out>...<end_working_out><SOLUTION>...</SOLUTION>`

2. **Mathematical Correctness** (40%): Is the solution mathematically sound and correct?
   - Compare with ground truth
   - Check logical reasoning steps

3. **Clarity and Completeness** (20%): Is the explanation clear and complete?
   - Are all steps explained?
   - Is the reasoning easy to follow?

4. **Conciseness** (10%): Is the solution appropriately concise without being verbose?

**Your Task:**
1. Evaluate both responses according to the criteria above
2. Provide a brief explanation for each model
3. Declare a winner (Model A, Model B, or Tie)

**Output Format (JSON):**
{{
  "model_a_scores": {{
    "format_adherence": <score 0-10>,
    "mathematical_correctness": <score 0-10>,
    "clarity_completeness": <score 0-10>,
    "conciseness": <score 0-10>,
    "total": <weighted total 0-10>
  }},
  "model_b_scores": {{
    "format_adherence": <score 0-10>,
    "mathematical_correctness": <score 0-10>,
    "clarity_completeness": <score 0-10>,
    "conciseness": <score 0-10>,
    "total": <weighted total 0-10>
  }},
  "model_a_feedback": "<brief feedback>",
  "model_b_feedback": "<brief feedback>",
  "winner": "<Model A|Model B|Tie>",
  "reason": "<1-2 sentence explanation of decision>"
}}

Respond ONLY with the JSON output, no other text."""

        # Call Bedrock
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.0,  # Deterministic for judging
            "messages": [
                {
                    "role": "user",
                    "content": judge_prompt
                }
            ]
        }

        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body)
            )

            response_body = json.loads(response['body'].read())
            judgment_text = response_body['content'][0]['text']

            # Parse JSON from response
            # Handle case where model might wrap JSON in markdown code blocks
            if "```json" in judgment_text:
                judgment_text = judgment_text.split("```json")[1].split("```")[0]
            elif "```" in judgment_text:
                judgment_text = judgment_text.split("```")[1].split("```")[0]

            judgment = json.loads(judgment_text.strip())
            return judgment

        except Exception as e:
            print(f"⚠️ Error calling Bedrock: {e}")
            # Return a default judgment in case of error
            return {
                "model_a_scores": {"total": 0},
                "model_b_scores": {"total": 0},
                "winner": "Error",
                "reason": f"Error during judgment: {str(e)}"
            }


def load_ground_truth(exercises_file: str = "exercises.jsonl") -> Dict[str, str]:
    """
    Load ground truth solutions from exercises.jsonl.

    Args:
        exercises_file: Path to exercises.jsonl

    Returns:
        Dictionary mapping exercise_text to solution_text
    """
    ground_truth = {}
    with open(exercises_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            ground_truth[data['exercise_text']] = data['solution_text']
    return ground_truth


def generate_report(
    results: List[Dict],
    base_model: str,
    sft_model: str,
    output_file: str
):
    """
    Generate markdown report of judgment results.

    Args:
        results: List of judgment results
        base_model: Name of base model
        sft_model: Name of fine-tuned model
        output_file: Output file path
    """
    # Calculate statistics
    base_wins = sum(1 for r in results if r['judgment']['winner'] == 'Model A')
    sft_wins = sum(1 for r in results if r['judgment']['winner'] == 'Model B')
    ties = sum(1 for r in results if r['judgment']['winner'] == 'Tie')
    errors = sum(1 for r in results if r['judgment']['winner'] == 'Error')

    total_problems = len(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# AI Judge Evaluation Report: Base vs Fine-tuned Model\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Model information
        f.write("## Models Evaluated\n\n")
        f.write(f"- **Base Model:** {base_model}\n")
        f.write(f"- **Fine-tuned Model:** {sft_model}\n")
        f.write(f"- **Judge Model:** AWS Bedrock (Claude 3.5 Sonnet)\n\n")

        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write(f"| Outcome | Count | Percentage |\n")
        f.write(f"|---------|-------|------------|\n")
        f.write(f"| **SFT Model Wins** | {sft_wins} | {sft_wins/total_problems*100:.1f}% |\n")
        f.write(f"| Base Model Wins | {base_wins} | {base_wins/total_problems*100:.1f}% |\n")
        f.write(f"| Ties | {ties} | {ties/total_problems*100:.1f}% |\n")
        if errors > 0:
            f.write(f"| Errors | {errors} | {errors/total_problems*100:.1f}% |\n")
        f.write(f"| **Total Problems** | {total_problems} | 100% |\n\n")

        # Key findings
        f.write("## Key Findings\n\n")
        if sft_wins > base_wins:
            margin = sft_wins - base_wins
            f.write(f"🎉 **Fine-tuned model outperformed base model** by {margin} problems ")
            f.write(f"({margin/total_problems*100:.1f}% margin).\n\n")
        elif base_wins > sft_wins:
            margin = base_wins - sft_wins
            f.write(f"⚠️ **Base model outperformed fine-tuned model** by {margin} problems ")
            f.write(f"({margin/total_problems*100:.1f}% margin). This is unexpected.\n\n")
        else:
            f.write(f"➖ **Models performed equally** with {sft_wins} wins each.\n\n")

        # Average scores
        avg_base_score = sum(r['judgment']['model_a_scores']['total'] for r in results) / total_problems
        avg_sft_score = sum(r['judgment']['model_b_scores']['total'] for r in results) / total_problems

        f.write("## Average Scores (0-10 scale)\n\n")
        f.write(f"- **Base Model:** {avg_base_score:.2f}\n")
        f.write(f"- **Fine-tuned Model:** {avg_sft_score:.2f}\n")
        f.write(f"- **Difference:** {avg_sft_score - avg_base_score:+.2f}\n\n")

        # Detailed results
        f.write("---\n\n")
        f.write("## Detailed Problem-by-Problem Results\n\n")

        for i, result in enumerate(results, 1):
            judgment = result['judgment']
            winner = judgment['winner']

            f.write(f"### Problem {i}\n\n")

            # Problem statement
            f.write(f"**Problem:**\n")
            f.write(f"```\n{result['problem'][:200]}...\n```\n\n")

            # Scores
            f.write(f"**Scores:**\n\n")
            f.write(f"| Model | Format | Math | Clarity | Concise | **Total** |\n")
            f.write(f"|-------|--------|------|---------|---------|-------|\n")

            model_a = judgment['model_a_scores']
            model_b = judgment['model_b_scores']

            f.write(f"| Base | {model_a.get('format_adherence', 'N/A')} | ")
            f.write(f"{model_a.get('mathematical_correctness', 'N/A')} | ")
            f.write(f"{model_a.get('clarity_completeness', 'N/A')} | ")
            f.write(f"{model_a.get('conciseness', 'N/A')} | ")
            f.write(f"**{model_a['total']:.1f}** |\n")

            f.write(f"| SFT | {model_b.get('format_adherence', 'N/A')} | ")
            f.write(f"{model_b.get('mathematical_correctness', 'N/A')} | ")
            f.write(f"{model_b.get('clarity_completeness', 'N/A')} | ")
            f.write(f"{model_b.get('conciseness', 'N/A')} | ")
            f.write(f"**{model_b['total']:.1f}** |\n\n")

            # Winner
            winner_emoji = "🟢" if winner == "Model B" else ("🔵" if winner == "Model A" else "⚪")
            f.write(f"**Winner:** {winner_emoji} {winner}\n\n")

            # Feedback
            f.write(f"**Base Model Feedback:** {judgment.get('model_a_feedback', 'N/A')}\n\n")
            f.write(f"**SFT Model Feedback:** {judgment.get('model_b_feedback', 'N/A')}\n\n")
            f.write(f"**Reason:** {judgment.get('reason', 'N/A')}\n\n")

            f.write("---\n\n")

        # Final summary
        f.write("## Conclusion\n\n")
        if sft_wins > base_wins:
            f.write(f"✅ The fine-tuned model demonstrates clear improvement over the base model, ")
            f.write(f"winning {sft_wins}/{total_problems} comparisons ({sft_wins/total_problems*100:.1f}%).\n")
        elif base_wins > sft_wins:
            f.write(f"⚠️ The fine-tuned model performed worse than the base model, ")
            f.write(f"winning only {sft_wins}/{total_problems} comparisons ({sft_wins/total_problems*100:.1f}%). ")
            f.write(f"This suggests the fine-tuning may need adjustment.\n")
        else:
            f.write(f"The models performed comparably, each winning {sft_wins}/{total_problems} comparisons.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Judge base vs SFT model comparison using AWS Bedrock"
    )
    parser.add_argument(
        "comparison_file",
        help="Path to comparison results JSON file"
    )
    parser.add_argument(
        "--exercises",
        default="exercises.jsonl",
        help="Path to exercises.jsonl (default: exercises.jsonl)"
    )
    parser.add_argument(
        "--model-id",
        default="anthropic.claude-3-5-sonnet-20241022-v2:0",
        help="Bedrock model ID for judge (default: Claude 3.5 Sonnet)"
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1)"
    )

    args = parser.parse_args()

    # Check if comparison file exists
    if not Path(args.comparison_file).exists():
        print(f"❌ Error: Comparison file not found: {args.comparison_file}")
        sys.exit(1)

    print("\n" + "="*80)
    print("🤖 AI JUDGE: Base vs SFT Model Evaluation")
    print("="*80)

    # Load comparison results
    print(f"\n📂 Loading comparison results from: {args.comparison_file}")
    with open(args.comparison_file, 'r', encoding='utf-8') as f:
        comparison_data = json.load(f)

    # Load ground truth
    print(f"📚 Loading ground truth from: {args.exercises}")
    ground_truth = load_ground_truth(args.exercises)

    # Initialize judge
    print(f"⚖️ Initializing AI judge: {args.model_id}")
    judge = BedrockJudge(model_id=args.model_id, region=args.region)

    # Get model names from metadata
    base_model = comparison_data['metadata']['base_model']
    sft_model = comparison_data['metadata']['sft_model']

    print(f"\n📊 Evaluating {len(comparison_data['results'])} problems...")
    print(f"   Base Model: {base_model}")
    print(f"   SFT Model: {sft_model}\n")

    # Judge each problem
    judged_results = []
    for i, result in enumerate(comparison_data['results'], 1):
        problem = result['problem']
        base_response = result['base_model']['response']
        sft_response = result['sft_model']['response']

        # Get ground truth for this problem
        truth = ground_truth.get(problem, "Ground truth not found")

        print(f"   Problem {i}/{len(comparison_data['results'])}...", end=' ')

        # Judge
        judgment = judge.judge_responses(
            problem=problem,
            ground_truth=truth,
            base_response=base_response,
            sft_response=sft_response
        )

        winner_symbol = "🟢" if judgment['winner'] == "Model B" else ("🔵" if judgment['winner'] == "Model A" else "⚪")
        print(f"{winner_symbol} {judgment['winner']}")

        judged_results.append({
            'problem': problem,
            'judgment': judgment
        })

    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"judge_report_{timestamp}.md"

    print(f"\n📝 Generating report: {report_file}")
    generate_report(
        results=judged_results,
        base_model=base_model,
        sft_model=sft_model,
        output_file=report_file
    )

    # Print summary to console
    base_wins = sum(1 for r in judged_results if r['judgment']['winner'] == 'Model A')
    sft_wins = sum(1 for r in judged_results if r['judgment']['winner'] == 'Model B')
    ties = sum(1 for r in judged_results if r['judgment']['winner'] == 'Tie')

    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"\n🟢 SFT Model Wins:  {sft_wins}/{len(judged_results)} ({sft_wins/len(judged_results)*100:.1f}%)")
    print(f"🔵 Base Model Wins: {base_wins}/{len(judged_results)} ({base_wins/len(judged_results)*100:.1f}%)")
    print(f"⚪ Ties:            {ties}/{len(judged_results)} ({ties/len(judged_results)*100:.1f}%)")

    if sft_wins > base_wins:
        print(f"\n🎉 Fine-tuned model is BETTER (+{sft_wins - base_wins} wins)")
    elif base_wins > sft_wins:
        print(f"\n⚠️ Base model is BETTER (+{base_wins - sft_wins} wins)")
    else:
        print(f"\n➖ Models performed EQUALLY")

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📄 View detailed report: {report_file}\n")


if __name__ == "__main__":
    main()
