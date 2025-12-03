"""
Memory-efficient comparison script for base vs SFT models.

This script handles memory constraints by:
1. Loading models sequentially (not simultaneously)
2. Clearing GPU memory between model loads
3. Saving outputs to files for comparison

Usage:
    python memory_efficient_comparison.py

Output:
    - comparison_results_TIMESTAMP.md: Human-readable markdown report
    - comparison_results_TIMESTAMP.json: Machine-readable JSON data
"""

import torch
import gc
import json
from datetime import datetime
from unsloth import FastLanguageModel

# Configuration
max_seq_length = 2048
model_path = "optimization_sft_model_v2"
# base_model_name = "unsloth/Qwen2.5-1.5B"
base_model_name = "unsloth/Qwen3-4B-Base_sft_model_v2"

# Reasoning tags
reasoning_start = "<start_working_out>"
reasoning_end = "<end_working_out>"
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

system_prompt = f"""You are given an optimization problem.
Think about the problem and provide your working out (proof steps).
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# Chat template
chat_template = \
    "{% if messages[0]['role'] == 'system' %}"\
        "{{ messages[0]['content'] + eos_token }}"\
        "{% set loop_messages = messages[1:] %}"\
    "{% else %}"\
        "{{ '{system_prompt}' + eos_token }}"\
        "{% set loop_messages = messages %}"\
    "{% endif %}"\
    "{% for message in loop_messages %}"\
        "{% if message['role'] == 'user' %}"\
            "{{ message['content'] }}"\
        "{% elif message['role'] == 'assistant' %}"\
            "{{ message['content'] + eos_token }}"\
        "{% endif %}"\
    "{% endfor %}"\
    "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"\
    "{% endif %}"

chat_template = chat_template\
    .replace("'{system_prompt}'", f"'{system_prompt}'")\
    .replace("'{reasoning_start}'", f"'{reasoning_start}'")


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
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

    return response


def test_model(model_name, is_base=True, test_problems=None, max_tokens=1024):
    """
    Test a model on given problems.

    Args:
        model_name: Model path or name
        is_base: Whether this is the base model (True) or fine-tuned (False)
        test_problems: List of problems to test
        max_tokens: Maximum tokens to generate

    Returns:
        List of results
    """
    print(f"\n{'='*80}")
    print(f"Loading {'BASE' if is_base else 'FINE-TUNED'} model: {model_name}")
    print(f"{'='*80}")

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        fast_inference=True,
    )

    # Apply chat template
    tokenizer.chat_template = chat_template

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    print(f"✅ Model loaded successfully")

    # Test on each problem
    results = []
    for i, problem in enumerate(test_problems):
        print(f"   Testing problem {i+1}/{len(test_problems)}...", end=' ')
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


def save_results_to_markdown(base_results, sft_results, filename):
    """Save comparison results to markdown file."""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Base vs SFT Model Comparison Results\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Models Compared:**\n")
        f.write(f"- Base Model: {base_model_name}\n")
        f.write(f"- Fine-tuned Model: {model_path}\n\n")
        f.write("---\n\n")

        # Detailed results
        for i, (base, sft) in enumerate(zip(base_results, sft_results)):
            f.write(f"## Problem {i+1}\n\n")

            f.write(f"### 📝 Problem Statement\n\n")
            f.write(f"```\n{base['problem']}\n```\n\n")

            f.write(f"### 🔵 BASE MODEL Response\n\n")
            f.write(f"```\n{base['response']}\n```\n\n")
            f.write(f"**Format Check:** {'✅ PASS' if base['format_check']['format_ok'] else '❌ FAIL'}\n")
            f.write(f"- Has `{reasoning_end}`: {base['format_check']['has_reasoning_end']}\n")
            f.write(f"- Has `{solution_start}`: {base['format_check']['has_solution_start']}\n")
            f.write(f"- Has `{solution_end}`: {base['format_check']['has_solution_end']}\n\n")

            f.write(f"### 🟢 SFT MODEL Response\n\n")
            f.write(f"```\n{sft['response']}\n```\n\n")
            f.write(f"**Format Check:** {'✅ PASS' if sft['format_check']['format_ok'] else '❌ FAIL'}\n")
            f.write(f"- Has `{reasoning_end}`: {sft['format_check']['has_reasoning_end']}\n")
            f.write(f"- Has `{solution_start}`: {sft['format_check']['has_solution_start']}\n")
            f.write(f"- Has `{solution_end}`: {sft['format_check']['has_solution_end']}\n\n")

            f.write(f"### 📈 Comparison Result\n\n")
            if sft['format_check']['format_ok'] and not base['format_check']['format_ok']:
                f.write("🎉 **SFT model successfully learned the format!**\n\n")
            elif base['format_check']['format_ok'] and sft['format_check']['format_ok']:
                f.write("✅ **Both models follow format** (compare proof quality manually)\n\n")
            elif not base['format_check']['format_ok'] and not sft['format_check']['format_ok']:
                f.write("⚠️ **Neither model follows format correctly**\n\n")
            else:
                f.write("⚠️ **Base model better than SFT** (unexpected)\n\n")

            f.write("---\n\n")

        # Summary statistics
        base_pass = sum(1 for r in base_results if r['format_check']['format_ok'])
        sft_pass = sum(1 for r in sft_results if r['format_check']['format_ok'])

        f.write("## 📊 Summary Statistics\n\n")
        f.write(f"| Metric | Base Model | SFT Model | Improvement |\n")
        f.write(f"|--------|------------|-----------|-------------|\n")
        f.write(f"| Correct Format | {base_pass}/{len(base_results)} ({base_pass/len(base_results)*100:.1f}%) | ")
        f.write(f"{sft_pass}/{len(sft_results)} ({sft_pass/len(sft_results)*100:.1f}%) | ")
        f.write(f"+{sft_pass - base_pass} ({(sft_pass - base_pass)/len(sft_results)*100:.1f}%) |\n\n")

        f.write("### Key Findings\n\n")
        if sft_pass > base_pass:
            f.write(f"✅ Fine-tuning **improved** format adherence by {sft_pass - base_pass} examples.\n")
        elif sft_pass == base_pass:
            f.write(f"➖ Both models performed **equally** on format adherence.\n")
        else:
            f.write(f"⚠️ Fine-tuning **decreased** format adherence (needs investigation).\n")


def save_results_to_json(base_results, sft_results, filename):
    """Save comparison results to JSON file."""
    data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'base_model': base_model_name,
            'sft_model': model_path,
            'num_problems': len(base_results)
        },
        'results': []
    }

    for i, (base, sft) in enumerate(zip(base_results, sft_results)):
        data['results'].append({
            'problem_id': i + 1,
            'problem': base['problem'],
            'base_model': {
                'response': base['response'],
                'format_check': base['format_check']
            },
            'sft_model': {
                'response': sft['response'],
                'format_check': sft['format_check']
            }
        })

    # Summary statistics
    base_pass = sum(1 for r in base_results if r['format_check']['format_ok'])
    sft_pass = sum(1 for r in sft_results if r['format_check']['format_ok'])

    data['summary'] = {
        'base_model_correct': base_pass,
        'sft_model_correct': sft_pass,
        'base_model_accuracy': base_pass / len(base_results),
        'sft_model_accuracy': sft_pass / len(sft_results),
        'improvement': sft_pass - base_pass
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """Main comparison workflow."""
    print("\n" + "="*80)
    print("🚀 BASE vs SFT MODEL COMPARISON")
    print("="*80)
    print(f"\nBase Model: {base_model_name}")
    print(f"Fine-tuned Model: {model_path}")

    # Load test problems from exercises.jsonl
    print("\n📚 Loading test problems...")
    import pandas as pd

    exercises = []
    with open("exercises.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            exercises.append(json.loads(line))

    df = pd.DataFrame(exercises)

    # Add length calculation (approximate)
    df['approx_length'] = df['exercise_text'].str.len() + df['solution_text'].str.len()
    df_sorted = df.sort_values('approx_length').reset_index(drop=True)

    # Take 10 smallest
    test_problems = df_sorted.iloc[:10]['exercise_text'].tolist()

    print(f"✅ Selected {len(test_problems)} smallest problems for testing\n")

    # Test BASE model
    print("\n" + "="*80)
    print("STEP 1: Testing BASE model")
    print("="*80)
    base_results = test_model(
        model_name=base_model_name,
        is_base=True,
        test_problems=test_problems,
        max_tokens=1024
    )

    # Test SFT model
    print("\n" + "="*80)
    print("STEP 2: Testing FINE-TUNED model")
    print("="*80)
    sft_results = test_model(
        model_name=model_path,
        is_base=False,
        test_problems=test_problems,
        max_tokens=1024
    )

    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_file = f"comparison_results_{timestamp}.md"
    json_file = f"comparison_results_{timestamp}.json"

    # Save results to files
    print("\n" + "="*80)
    print("STEP 3: Saving results to files")
    print("="*80)

    print(f"📝 Saving markdown report to: {markdown_file}")
    save_results_to_markdown(base_results, sft_results, markdown_file)

    print(f"💾 Saving JSON data to: {json_file}")
    save_results_to_json(base_results, sft_results, json_file)

    # Print summary to console
    base_pass = sum(1 for r in base_results if r['format_check']['format_ok'])
    sft_pass = sum(1 for r in sft_results if r['format_check']['format_ok'])

    print("\n" + "="*80)
    print("📊 SUMMARY STATISTICS")
    print("="*80)
    print(f"\nBase Model: {base_pass}/{len(base_results)} correct format ({base_pass/len(base_results)*100:.1f}%)")
    print(f"SFT Model:  {sft_pass}/{len(sft_results)} correct format ({sft_pass/len(sft_results)*100:.1f}%)")
    print(f"\nImprovement: +{sft_pass - base_pass} examples ({(sft_pass - base_pass)/len(sft_results)*100:.1f}%)")

    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📄 View detailed results in: {markdown_file}")
    print(f"📄 View JSON data in: {json_file}\n")


if __name__ == "__main__":
    main()
