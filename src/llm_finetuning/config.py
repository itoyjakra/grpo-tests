"""Configuration dataclasses for LLM fine-tuning pipeline.

This module defines all configuration parameters for model loading, training,
and evaluation. Configurations can be loaded from YAML files or instantiated
programmatically.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for model loading and LoRA setup.

    Attributes:
        model_name: HuggingFace model identifier or local path
        max_seq_length: Maximum sequence length for training
        lora_rank: Rank for LoRA adaptation (higher = more parameters)
        lora_alpha: LoRA scaling factor (typically rank * 2)
        load_in_4bit: Whether to load model in 4-bit quantization
        fast_inference: Enable vLLM for fast inference
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        target_modules: List of module names to apply LoRA to
        use_gradient_checkpointing: Enable gradient checkpointing for memory efficiency
        random_state: Random seed for reproducibility
    """
    model_name: str = "unsloth/Qwen3-4B-Base"
    max_seq_length: int = 2048
    lora_rank: int = 32
    lora_alpha: int = 64  # rank * 2
    load_in_4bit: bool = False
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.9
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class DataConfig:
    """Configuration for dataset loading and processing.

    Attributes:
        sft_dataset_name: Dataset name/path for SFT pre-training
        sft_dataset_split: Split to use for SFT dataset
        grpo_dataset_name: Dataset name/path for GRPO training
        grpo_dataset_split: Split to use for GRPO dataset
        grpo_dataset_config: Configuration name for GRPO dataset
        max_length_percentile: Percentile for filtering long sequences (0-100)
        truncate_ratio: Ratio of max_seq_length for truncation (0.0-1.0)
        filter_numeric_only: Filter to only numeric answers for SFT
    """
    sft_dataset_name: str = "unsloth/OpenMathReasoning-mini"
    sft_dataset_split: str = "cot"
    grpo_dataset_name: str = "open-r1/DAPO-Math-17k-Processed"
    grpo_dataset_split: str = "train"
    grpo_dataset_config: str = "en"
    max_length_percentile: int = 90
    truncate_ratio: float = 0.5
    filter_numeric_only: bool = True


@dataclass
class PromptConfig:
    """Configuration for prompt templates and formatting tags.

    Attributes:
        reasoning_start: Start tag for reasoning/thinking section
        reasoning_end: End tag for reasoning/thinking section
        solution_start: Start tag for final solution
        solution_end: End tag for final solution
        system_prompt_template: Template for system prompt message
    """
    reasoning_start: str = "<start_working_out>"
    reasoning_end: str = "<end_working_out>"
    solution_start: str = "<SOLUTION>"
    solution_end: str = "</SOLUTION>"
    system_prompt_template: str = (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        "Place it between {reasoning_start} and {reasoning_end}.\n"
        "Then, provide your solution between {solution_start}{solution_end}"
    )


@dataclass
class SFTTrainingConfig:
    """Configuration for Supervised Fine-Tuning.

    Attributes:
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of steps to accumulate gradients
        warmup_steps: Number of warmup steps for learning rate scheduler
        num_train_epochs: Number of training epochs
        learning_rate: Peak learning rate
        logging_steps: Log metrics every N steps
        optim: Optimizer name (e.g., 'adamw_8bit', 'adamw_torch')
        weight_decay: Weight decay for regularization
        lr_scheduler_type: Learning rate scheduler type
        seed: Random seed for reproducibility
        output_dir: Directory to save checkpoints
        save_steps: Save checkpoint every N steps
        report_to: Experiment tracking backend ('none', 'wandb', 'tensorboard')
    """
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 5
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    logging_steps: int = 5
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs/sft"
    save_steps: int = 100
    report_to: str = "none"


@dataclass
class VLLMSamplingConfig:
    """Configuration for vLLM sampling parameters.

    Attributes:
        temperature: Sampling temperature (0.0 = greedy, higher = more random)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling (-1 = disabled)
        min_p: Minimum probability threshold
        seed: Random seed for sampling
        include_stop_str_in_output: Include stop string in output
    """
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.1
    seed: int = 3407
    include_stop_str_in_output: bool = True


@dataclass
class GRPOTrainingConfig:
    """Configuration for Group Relative Policy Optimization training.

    Attributes:
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of steps to accumulate gradients
        num_generations: Number of completions to generate per prompt
        max_steps: Maximum number of training steps
        learning_rate: Learning rate (typically much lower than SFT)
        weight_decay: Weight decay for regularization
        warmup_ratio: Fraction of steps for warmup
        lr_scheduler_type: Learning rate scheduler type
        optim: Optimizer name
        logging_steps: Log metrics every N steps
        save_steps: Save checkpoint every N steps
        output_dir: Directory to save checkpoints
        report_to: Experiment tracking backend
        eval_strategy: Evaluation strategy ('no', 'steps', 'epoch')
        eval_steps: Evaluate every N steps
    """
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_generations: int = 4
    max_steps: int = 100
    learning_rate: float = 5e-6
    weight_decay: float = 0.001
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    save_steps: int = 100
    output_dir: str = "outputs/grpo"
    report_to: str = "none"
    eval_strategy: str = "no"
    eval_steps: Optional[int] = None


@dataclass
class RewardConfig:
    """Configuration for reward functions.

    Attributes:
        use_format_exact: Enable exact format matching reward
        use_format_approximate: Enable approximate format matching reward
        use_answer_check: Enable answer correctness reward
        use_number_check: Enable numerical answer reward
        print_every_steps: Print debug info every N steps (0 = never)
    """
    use_format_exact: bool = True
    use_format_approximate: bool = True
    use_answer_check: bool = True
    use_number_check: bool = True
    print_every_steps: int = 0


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation.

    Attributes:
        num_test_problems: Number of test problems to evaluate
        temperature: Sampling temperature for generation
        top_k: Top-k sampling for generation
        max_new_tokens: Maximum tokens to generate
        test_problems: List of custom test problems (optional)
    """
    num_test_problems: int = 5
    temperature: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 2048
    test_problems: Optional[List[str]] = None


@dataclass
class PipelineConfig:
    """Complete pipeline configuration combining all sub-configs.

    Attributes:
        model: Model configuration
        data: Data configuration
        prompt: Prompt configuration
        sft_training: SFT training configuration
        grpo_training: GRPO training configuration
        vllm_sampling: vLLM sampling configuration
        reward: Reward function configuration
        evaluation: Evaluation configuration
        skip_sft: Skip SFT pre-training phase
        skip_grpo: Skip GRPO training phase
        save_path: Path to save final model
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    sft_training: SFTTrainingConfig = field(default_factory=SFTTrainingConfig)
    grpo_training: GRPOTrainingConfig = field(default_factory=GRPOTrainingConfig)
    vllm_sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    skip_sft: bool = False
    skip_grpo: bool = False
    save_path: str = "grpo_saved_lora"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PipelineConfig instance with loaded parameters

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML is malformed
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            prompt=PromptConfig(**config_dict.get('prompt', {})),
            sft_training=SFTTrainingConfig(**config_dict.get('sft_training', {})),
            grpo_training=GRPOTrainingConfig(**config_dict.get('grpo_training', {})),
            vllm_sampling=VLLMSamplingConfig(**config_dict.get('vllm_sampling', {})),
            reward=RewardConfig(**config_dict.get('reward', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            skip_sft=config_dict.get('skip_sft', False),
            skip_grpo=config_dict.get('skip_grpo', False),
            save_path=config_dict.get('save_path', 'grpo_saved_lora'),
        )

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path where YAML file will be saved
        """
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'prompt': self.prompt.__dict__,
            'sft_training': self.sft_training.__dict__,
            'grpo_training': self.grpo_training.__dict__,
            'vllm_sampling': self.vllm_sampling.__dict__,
            'reward': self.reward.__dict__,
            'evaluation': self.evaluation.__dict__,
            'skip_sft': self.skip_sft,
            'skip_grpo': self.skip_grpo,
            'save_path': self.save_path,
        }

        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
