# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM fine-tuning pipeline that trains language models using a two-phase approach: **Supervised Fine-Tuning (SFT)** followed by **Group Relative Policy Optimization (GRPO)**. The pipeline is built on Unsloth for efficient training and vLLM for fast inference during GRPO.

The goal is to fine-tune models (default: Qwen3-4B) to solve mathematical reasoning problems with a specific output format that includes structured reasoning steps and final solutions.

## Environment Setup

This project uses **uv** as the package manager. The virtual environment is in `.venv/`.

### Dependencies

Key dependencies defined in `pyproject.toml`:
- `torch-c-dlpack-ext>=0.1.3`
- `unsloth>=2025.11.3` - Efficient LoRA training
- `vllm>=0.11.2` - Fast parallel inference for GRPO

All other dependencies (transformers, datasets, etc.) are installed automatically as transitive dependencies.

### Activate Environment

```bash
# Activate virtual environment (if needed for manual commands)
source .venv/bin/activate

# Or use uv run to execute commands in the environment
uv run python <script.py>
```

## Running the Pipeline

### Basic Commands

```bash
# Full pipeline (SFT + GRPO)
uv run python run_training.py

# With custom config
uv run python run_training.py --config configs/my_config.yaml

# Skip SFT (e.g., if already trained)
uv run python run_training.py --skip-sft

# Skip GRPO (SFT only)
uv run python run_training.py --skip-grpo

# Override specific parameters
uv run python run_training.py --max-steps 200 --learning-rate 1e-5

# For GPU memory optimization (if hitting OOM errors)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run_training.py
```

### Monitor Training

Training logs are written to `training.log` in real-time:

```bash
tail -f training.log
```

## Architecture Overview

### Pipeline Flow

The training pipeline (`run_training.py`) orchestrates three phases:

1. **Phase 0: Model Loading**
   - Load base model with LoRA configuration
   - Configure custom chat template with reasoning/solution tags
   - Managed by `ModelManager` (`src/llm_finetuning/model.py`)

2. **Phase 1: SFT Pre-Training**
   - Teaches model the custom output format
   - Dataset: OpenMathReasoning-mini with chain-of-thought examples
   - Formats examples with custom tags: `<start_working_out>...<end_working_out><SOLUTION>...</SOLUTION>`
   - Managed by `SFTTrainingPipeline` (`src/llm_finetuning/training/sft_trainer.py`)

3. **Phase 2: GRPO Training**
   - Reinforcement learning phase using reward functions
   - Dataset: DAPO-Math-17k-Processed
   - Generates multiple completions per prompt using vLLM
   - Scores completions with 4 reward functions (format matching, answer correctness)
   - Managed by `GRPOTrainingPipeline` (`src/llm_finetuning/training/grpo_trainer.py`)

### Key Components

**Configuration System** (`src/llm_finetuning/config.py`):
- All parameters defined in dataclasses
- Load from YAML: `PipelineConfig.from_yaml("configs/default_config.yaml")`
- 9 config classes: `ModelConfig`, `DataConfig`, `PromptConfig`, `SFTTrainingConfig`, `GRPOTrainingConfig`, `VLLMSamplingConfig`, `RewardConfig`, `EvaluationConfig`, `PipelineConfig`

**Data Processing** (`src/llm_finetuning/data/`):
- `DatasetLoader`: Loads HuggingFace datasets
- `DatasetProcessor`: Formats, tokenizes, filters by length
- `PromptTemplateManager`: Manages Jinja2 chat templates and custom tags

**Reward Functions** (`src/llm_finetuning/rewards/reward_functions.py`):
- `match_format_exactly()`: +3.0 for exact format match
- `match_format_approximately()`: -3.0 to +1.5 based on tag presence
- `check_answer()`: -2.0 to +5.0 based on solution correctness
- `check_numbers()`: -1.5 to +3.0 based on numerical matching
- Total reward range: -6.5 to +12.5

**Model Management** (`src/llm_finetuning/model.py`):
- `ModelManager`: Loads models with Unsloth, applies LoRA, saves weights

## Configuration Details

### Default Config Location

`configs/default_config.yaml` - contains all default parameters

### Critical Parameters for GPU Memory

If you encounter CUDA OOM errors, adjust these in the config:

```yaml
model:
  gpu_memory_utilization: 0.7  # Reduce if OOM (default: 0.7)

grpo_training:
  num_generations: 2  # Reduce if OOM (default: 2, can go to 1)
  gradient_accumulation_steps: 2  # Increase if OOM (default: 2)
  per_device_train_batch_size: 1  # Already minimal
```

### Dataset Column Requirements

**SFT Dataset** must have:
- `expected_answer`: The correct answer
- `problem`: The math problem
- `generated_solution`: Chain-of-thought reasoning

**GRPO Dataset** must have:
- `prompt`: The question/problem
- `solution` or `answer`: The correct answer (processor auto-renames `solution` → `answer`)

## Important Implementation Details

### Custom Prompt Format

The pipeline teaches models to output in this format:

```
<start_working_out>
[reasoning steps here]
<end_working_out>
<SOLUTION>
[final answer]
</SOLUTION>
```

Tags are configurable in `PromptConfig` and injected into the chat template via Jinja2.

### GRPO Dataset Processing

The processor calculates `max_prompt_length` based on percentile filtering (default 90th percentile) to ensure prompts + completions fit within `max_seq_length`. This prevents truncation during generation.

Key code: `src/llm_finetuning/data/processor.py:130-215`

### Memory Management

Between SFT and GRPO phases, the pipeline calls:
```python
torch.cuda.empty_cache()
gc.collect()
```

This is critical because vLLM (used in GRPO) needs substantial GPU memory for KV cache.

### vLLM Integration

GRPO uses vLLM for fast parallel generation of multiple completions per prompt. The vLLM engine is configured with:
- `SamplingParams`: temperature, top_p, top_k, min_p
- LoRA adapters loaded from the SFT checkpoint
- GPU memory allocation separate from training memory

### Checkpoint and Output Structure

**SFT Checkpoints** (`outputs/sft/checkpoint-*/`):
- `adapter_config.json` - LoRA adapter configuration
- `adapter_model.safetensors` - LoRA weight matrices
- `chat_template.jinja` - Custom chat template with reasoning tags
- Tokenizer files (`tokenizer.json`, `vocab.json`, `merges.txt`, etc.)
- Training state (`optimizer.pt`, `scheduler.pt`, `trainer_state.json`)

**GRPO Checkpoints** (`outputs/grpo/checkpoint-*/`):
- Same structure as SFT checkpoints
- Contains LoRA adapters fine-tuned with GRPO

**Final Model** (`grpo_saved_lora/`):
- Contains only the final LoRA adapter weights
- `adapter_config.json` and `adapter_model.safetensors`
- No optimizer/scheduler state (smaller size for inference)

## Common Issues and Solutions

### Bug: Missing 'answer' field
**Error**: `ValueError: Training dataset must have 'answer' field`
**Solution**: Dataset has 'solution' column - processor automatically renames it (line 210-213 in `processor.py`)

### Bug: CUDA Out of Memory
**Error**: `CUDA out of memory`
**Solution**:
1. Reduce `gpu_memory_utilization` from 0.9 to 0.7
2. Reduce `num_generations` from 4 to 2
3. Increase `gradient_accumulation_steps` from 1 to 2
4. Use env var: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Reward Values Stay at -6.5
This is **expected** at the start of GRPO training. The model hasn't learned the format yet. Rewards will improve as training progresses. Typical progression:
- Step 0-20: -6.5 (no format, wrong answers)
- Step 20-50: -3.0 to 0.0 (partial format learning)
- Step 50-100: 0.0 to +5.0 (format learned, improving answers)

## Evaluation System

### Overview

The pipeline includes a comprehensive evaluation system that monitors model performance during GRPO training.

### Enable Evaluation (Default: Enabled)

Evaluation is configured in `configs/default_config.yaml`:

```yaml
grpo_training:
  eval_strategy: "steps"          # Evaluate periodically
  eval_steps: 20                  # Evaluate every 20 steps
  train_validation_split: 0.9     # 90% train, 10% validation
```

### Evaluation Metrics

During training, the system tracks:
- **Format Accuracy**: Percentage of outputs with correct reasoning/solution tags
- **Answer Accuracy**: Percentage of correct final answers
- **Reasoning Present**: Percentage with reasoning content
- **Avg Reasoning Length**: Average character count of reasoning sections
- **Avg Total Reward**: Average reward from reward functions

### Expected Metric Progression

- **Step 0** (SFT baseline): Format ~85%, Answer ~50%, Reward ~4.0
- **Mid-training** (Step 50): Format ~95%, Answer ~65%, Reward ~7.5
- **End-training** (Step 100): Format ~98%, Answer ~70%, Reward ~9.0

### How It Works

1. Dataset automatically splits into 90% training / 10% validation
2. Every `eval_steps`, model evaluated on validation set
3. Metrics logged to console and `training.log`
4. Compare Step 0 (SFT-only) vs final step (after GRPO) to measure improvement

For detailed guidance, see `EVALUATION_GUIDE.md`.

## Debugging Tools

### Debug GRPO Generations

If reward variance is zero or generations are identical, use the debug script:

```bash
uv run python debug_grpo.py
```

This script:
- Loads the SFT checkpoint
- Generates 5 completions with GRPO sampling settings
- Checks for format tags and diversity
- Helps diagnose issues with generation parameters

Common issues diagnosed:
- Temperature too low (generations too similar)
- Missing format tags after SFT
- Tokenization problems

## Testing and Debugging

### Verify Installation
```bash
# Check CUDA availability
uv run python check.py

# Test module imports without training
uv run python test_modules.py
```

### Test Evaluation System
```bash
# Verify evaluation metrics calculation
uv run python test_evaluation.py
```

### Syntax Verification
```bash
# Check Python syntax across all modules
python test_syntax.py
```

## File Structure

```
grpo-tests/
├── run_training.py           # Main pipeline orchestrator
├── main.py                   # Simple entry point (legacy)
├── check.py                  # CUDA availability checker
├── test_modules.py           # Module import tests
├── test_syntax.py            # Syntax verification
├── configs/
│   └── default_config.yaml   # Default configuration
├── src/llm_finetuning/
│   ├── __init__.py           # Module exports (factory functions)
│   ├── config.py             # Configuration dataclasses
│   ├── model.py              # Model loading/saving
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py         # Dataset loading
│   │   ├── processor.py      # Dataset processing & formatting
│   │   └── templates.py      # Prompt template management
│   ├── rewards/
│   │   ├── __init__.py
│   │   └── reward_functions.py  # GRPO reward functions
│   ├── training/
│   │   ├── __init__.py
│   │   ├── sft_trainer.py    # SFT training pipeline
│   │   └── grpo_trainer.py   # GRPO training pipeline
│   ├── evaluation/           # Placeholder for evaluation
│   └── utils/                # Placeholder for utilities
└── outputs/                  # Training outputs
    ├── sft/                  # SFT checkpoints
    │   ├── checkpoint-*/     # Intermediate checkpoints
    │   └── final/            # Final SFT model
    └── grpo/                 # GRPO checkpoints
        ├── checkpoint-*/     # Intermediate checkpoints
        └── final/            # Final GRPO model
```

## Development Notes

### Type Checking
All code uses comprehensive type hints. Key types:
- `PreTrainedModel`, `PreTrainedTokenizer` from `transformers`
- `Dataset` from `datasets`
- `SamplingParams`, `GRPOConfig`, `GRPOTrainer` from `unsloth`

### Factory Pattern
All components use factory functions for instantiation:
- `create_model_manager()`
- `create_dataset_loader()`
- `create_dataset_processor()`
- `create_reward_manager()`
- `create_sft_trainer()`
- `create_grpo_trainer()`

This provides a consistent interface and makes testing easier.

### Logging
All modules use Python's `logging` module. Logger names match module paths. Set `logging_steps` in config to control frequency.

## Expected Training Time

On an A100 40GB GPU:
- **SFT**: ~30-40 minutes (2 epochs, ~118 steps)
- **GRPO**: ~20-30 minutes (100 steps at ~18s/step)
- **Total**: ~1-1.5 hours for full pipeline

Final model is saved to `grpo_saved_lora/` (configurable via `save_path` in config).

## Using the Trained Model

The final LoRA adapter is saved in `grpo_saved_lora/`. To use it for inference:

```python
from unsloth import FastLanguageModel

# Load base model + LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-4B-Base",
    adapter_name="grpo_saved_lora",  # Load your trained adapter
    max_seq_length=2048,
    load_in_4bit=False,
)

# Use the custom chat template
messages = [
    {"role": "user", "content": "What is 15 * 23?"}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Expected output format:
```
<start_working_out>
15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345
<end_working_out>
<SOLUTION>
345
</SOLUTION>
```
