"""Chat templates and prompt formatting for LLM fine-tuning.

This module handles system prompts, chat templates, and message formatting
for the training pipeline.
"""

from typing import Dict, Tuple
from transformers import PreTrainedTokenizer

from ..config import PromptConfig


class PromptTemplateManager:
    """Manages prompt templates and chat template configuration.

    This class handles the creation of system prompts and configuration
    of tokenizer chat templates with custom formatting tags.

    Attributes:
        config: Prompt configuration with formatting tags
        system_prompt: Formatted system prompt string
        prompt_components: Dictionary of formatting tag components
    """

    def __init__(self, config: PromptConfig):
        """Initialize prompt template manager.

        Args:
            config: Prompt configuration with tags and templates
        """
        self.config = config
        self.system_prompt = self._create_system_prompt()
        self.prompt_components = self._get_prompt_components()

    def _create_system_prompt(self) -> str:
        """Create system prompt from template.

        Returns:
            Formatted system prompt string with custom tags
        """
        return self.config.system_prompt_template.format(
            reasoning_start=self.config.reasoning_start,
            reasoning_end=self.config.reasoning_end,
            solution_start=self.config.solution_start,
            solution_end=self.config.solution_end
        )

    def _get_prompt_components(self) -> Dict[str, str]:
        """Get dictionary of prompt components.

        Returns:
            Dictionary mapping component names to their values
        """
        return {
            "reasoning_start": self.config.reasoning_start,
            "reasoning_end": self.config.reasoning_end,
            "solution_start": self.config.solution_start,
            "solution_end": self.config.solution_end,
        }

    def configure_chat_template(self, tokenizer: PreTrainedTokenizer) -> str:
        """Configure tokenizer with custom chat template.

        The chat template is configured to:
        1. Handle system/user/assistant roles
        2. Add EOS tokens appropriately
        3. Insert reasoning_start tag when generating

        Args:
            tokenizer: Tokenizer to configure

        Returns:
            The configured chat template string
        """
        chat_template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{{ '{system_prompt}' + eos_token }}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
            "{% endif %}"
        )

        # Replace placeholders with actual values
        chat_template = chat_template.replace(
            "'{system_prompt}'", f"'{self.system_prompt}'"
        ).replace(
            "'{reasoning_start}'", f"'{self.config.reasoning_start}'"
        )

        tokenizer.chat_template = chat_template
        return chat_template

    def get_system_prompt(self) -> str:
        """Get the system prompt.

        Returns:
            System prompt string
        """
        return self.system_prompt

    def get_components(self) -> Dict[str, str]:
        """Get prompt components dictionary.

        Returns:
            Dictionary of formatting tags
        """
        return self.prompt_components


def create_prompt_manager(config: PromptConfig) -> PromptTemplateManager:
    """Factory function to create a prompt template manager.

    Args:
        config: Prompt configuration

    Returns:
        Initialized PromptTemplateManager instance
    """
    return PromptTemplateManager(config)
