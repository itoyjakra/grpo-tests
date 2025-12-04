"""
Test third-party model separately and save results to JSON.

This script tests a third-party model on optimization problems and saves
results to a JSON file that can be merged with base/SFT results during judging.

Usage:
    python test_third_party_model.py --model ant-opt/LLMOPT-Qwen2.5-14B
    python test_third_party_model.py --model ant-opt/LLMOPT-Qwen2.5-14B --num-problems 10
"""

import torch
import gc
import json
import argparse
from datetime import datetime
from pathlib import Path
from unsloth import FastLanguageModel
import pandas as pd
import html
import re

# Configuration
max_seq_length = 2048

# Reasoning tags
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given an optimization problem.
Think about the problem and provide your working out (proof steps).
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""


def clear_memory():
    """Clear GPU memory aggressively."""
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()

    gc.collect()
    print("✅ GPU memory cleared")


def check_format(response):
    """Check if response has correct format."""
    has_reasoning_end = reasoning_end in response
    has_solution_start = solution_start in response
    has_solution_end = solution_end in response
    return {
        'has_reasoning_end': has_reasoning_end,
        'has_solution_start': has_solution_start,
        'has_solution_end': has_solution_end,
        'format_ok': all([has_reasoning_end, has_solution_start, has_solution_end])
    }


def clean_response(response):
    """Clean up response from third-party models."""
    # Unescape HTML entities
    response = html.unescape(response)

    # Fix excessive backslash escaping (e.g., \\\\\\\\ -> \\)
    response = re.sub(r'\\{4,}', r'\\\\', response)

    # Extract the final answer if it's at the end without proper tags
    if solution_start not in response and solution_end not in response:
        answer_patterns = [
            r'The final answer is[:\s]+(.+?)(?:\.|$)',
            r'Therefore[,\s]+(.+?)(?:\.|$)',
            r'Thus[,\s]+(.+?)(?:\.|$)',
            r'The (?:dual cone|answer|solution) is[:\s]+(.+?)(?:\.|$)',
            r'In conclusion[,\s]+(.+?)(?:\.|$)',
        ]

        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                answer_text = match.group(1).strip()
                # Wrap it in solution tags
                response = response + f"\n{solution_start}\n{answer_text}\n{solution_end}"
                break

    return response


def generate_response(model, tokenizer, problem, max_tokens=1024):
    """Generate response for a given problem."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    # Apply cleanup for third-party models
    response = clean_response(response)

    return response


def test_third_party_model(model_name, test_problems, max_tokens=1024):
    """
    Test a third-party model on given problems.

    Args:
        model_name: Model path or name
        test_problems: List of problems to test
        max_tokens: Maximum tokens to generate

    Returns:
        List of results
    """
    print(f"\n{'='*80}")
    print(f"Loading third-party model: {model_name}")
    print(f"   Using 4-bit quantization to save memory")
    print(f"{'='*80}")

    # Load model with 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,
    )

    print(f"   Using model's default chat template (no custom tags)")

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    print(f"✅ Model loaded successfully\n")

    # Test on each problem
    results = []
    for i, problem in enumerate(test_problems):
        print(f"   Testing problem {i+1}/{len(test_problems)}...", end=' ', flush=True)
        response = generate_response(model, tokenizer, problem, max_tokens)
        format_check = check_format(response)

        results.append({
            'problem': problem,
            'response': response,
            'format_check': format_check
        })

        print(f"{'✅' if format_check['format_ok'] else '❌'}")

    # Clean up
    del model
    del tokenizer
    clear_memory()

    return results


def save_results_to_json(results, model_name, filename):
    """Save third-party model results to JSON file."""
    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'num_problems': len(results)
        },
        'results': []
    }

    for i, result in enumerate(results):
        data['results'].append({
            'problem_id': i + 1,
            'problem': result['problem'],
            'response': result['response'],
            'format_check': result['format_check']
        })

    # Summary statistics
    pass_count = sum(1 for r in results if r['format_check']['format_ok'])

    data['summary'] = {
        'correct_format': pass_count,
        'accuracy': pass_count / len(results),
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main workflow."""
    parser = argparse.ArgumentParser(
        description="Test third-party model and save results to JSON"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Third-party model to test (e.g., ant-opt/LLMOPT-Qwen2.5-14B)"
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help="Number of problems to test (default: 10)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024)"
    )
    parser.add_argument(
        "--output",
        help="Output JSON filename (default: third_party_results_TIMESTAMP.json)"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 THIRD-PARTY MODEL TESTING")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Number of problems: {args.num_problems}")
    print(f"Max tokens: {args.max_tokens}")

    # Load test problems from exercises.jsonl
    print("\n📚 Loading test problems...")

    exercises = []
    with open("exercises.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            exercises.append(json.loads(line))

    df = pd.DataFrame(exercises)

    # Add length calculation (approximate)
    df['approx_length'] = df['exercise_text'].str.len() + df['solution_text'].str.len()
    df_sorted = df.sort_values('approx_length').reset_index(drop=True)

    # Take N smallest problems
    test_problems = df_sorted.iloc[:args.num_problems]['exercise_text'].tolist()

    print(f"✅ Selected {len(test_problems)} smallest problems for testing\n")

    # Test third-party model
    results = test_third_party_model(
        model_name=args.model,
        test_problems=test_problems,
        max_tokens=args.max_tokens
    )

    # Generate output filename
    if args.output:
        json_file = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize model name for filename
        model_slug = args.model.replace('/', '_').replace('.', '_')
        json_file = f"third_party_results_{model_slug}_{timestamp}.json"

    # Save results
    print("\n" + "="*80)
    print("💾 SAVING RESULTS")
    print("="*80)

    print(f"Saving to: {json_file}")
    save_results_to_json(results, args.model, json_file)

    # Print summary
    pass_count = sum(1 for r in results if r['format_check']['format_ok'])

    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    print(f"\nCorrect format: {pass_count}/{len(results)} ({pass_count/len(results)*100:.1f}%)")

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📄 Results saved: {json_file}")
    print(f"\n💡 Next steps:")
    print(f"   1. Run base/SFT comparison to generate base_sft_results.json")
    print(f"   2. Run judge script with both JSON files\n")


if __name__ == "__main__":
    main()
