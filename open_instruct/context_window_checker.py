"""
Context window checking utilities for litellm acompletion calls.

This module provides functions to check if acompletion calls would exceed the context window
before making the API request, preventing ContextWindowExceededError.

Usage:
    from open_instruct.context_window_checker import (
        check_context_window_limit,
        truncate_messages_to_fit_context,
        safe_acompletion_with_context_check
    )

    # Check before making a call
    if check_context_window_limit(messages, max_tokens, model_name):
        response = await acompletion(...)
    else:
        # Handle context window exceeded
        pass
"""

import os
from functools import lru_cache

import tiktoken
from transformers import AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

# By default avoid remote HuggingFace tokenizer fetches in the hot path.
# Enable HF tokenizer loading explicitly if needed for strict token accounting.
USE_HF_TOKENIZER = os.getenv("RLVR_CONTEXT_WINDOW_USE_HF_TOKENIZER", "false").lower() in {
    "1",
    "true",
    "yes",
}
HF_TOKENIZER_LOCAL_ONLY = os.getenv("RLVR_CONTEXT_WINDOW_HF_LOCAL_ONLY", "true").lower() in {
    "1",
    "true",
    "yes",
}


def _normalize_model_name(model_name: str) -> str:
    return model_name.replace("hosted_vllm/", "")


@lru_cache(maxsize=32)
def _get_hf_tokenizer(model_name: str):
    if not USE_HF_TOKENIZER:
        return None
    normalized_model_name = _normalize_model_name(model_name)
    try:
        return AutoTokenizer.from_pretrained(
            normalized_model_name,
            local_files_only=HF_TOKENIZER_LOCAL_ONLY,
        )
    except Exception as e:
        logger.warning(
            f"Failed to load HuggingFace tokenizer for {model_name}: {e}. Falling back to tiktoken."
        )
        return None


def get_encoding_for_model(model_name: str):
    """
    Get the appropriate tiktoken encoding for a given model.

    Args:
        model_name: Name of the model

    Returns:
        tiktoken.Encoding: The appropriate encoding for the model
    """
    model_name_lower = model_name.lower()

    # GPT models use cl100k_base
    if "gpt-4" in model_name_lower or "gpt-3.5" in model_name_lower or "claude" in model_name_lower:
        return tiktoken.get_encoding("cl100k_base")

    # Models that use gpt2 encoding (including OLMo and other AI2 models)
    gpt2_models = [
        "qwen",
        "llama",
        "llama2",
        "llama3",
        "mistral",
        "codellama",
        "olmo",
        "olmoe",
        "olmo2",  # AI2 OLMo models use GPTNeoX tokenizer (GPT-2 based)
        "allenai/olmo",
        "allenai/olmoe",
        "allenai/olmo2",  # Full model names
    ]
    if any(model in model_name_lower for model in gpt2_models):
        return tiktoken.get_encoding("gpt2")

    # Default to cl100k_base for unknown models
    else:
        logger.warning(f"Unknown model {model_name}, defaulting to cl100k_base encoding")
        return tiktoken.get_encoding("cl100k_base")


def check_context_window_limit(
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    model_name: str,
    max_context_length: int = 8192,
    safety_margin: int = 100,
) -> bool:
    """
    Check if the final answer length would exceed the context window.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        max_completion_tokens: Maximum tokens for completion
        model_name: Name of the model (used to determine tokenizer)
        max_context_length: Maximum context length for the model
        safety_margin: Additional safety margin in tokens

    Returns:
        bool: True if the request would fit within context window, False otherwise
    """
    # First try to use a cached HuggingFace tokenizer if enabled/available.
    tokenizer = _get_hf_tokenizer(model_name)
    if tokenizer is not None:
        try:
            max_context_length = tokenizer.model_max_length if max_context_length is None else max_context_length

            total_message_tokens = 0
            for message in messages:
                content = message.get("content", "")
                role = message.get("role", "")
                content_tokens = len(tokenizer.encode(content, add_special_tokens=False))
                role_tokens = 4 if role == "system" else 3
                total_message_tokens += content_tokens + role_tokens

            total_tokens_needed = total_message_tokens + max_completion_tokens + safety_margin
            if total_tokens_needed > max_context_length:
                logger.warning(
                    f"Judge context window would be exceeded: {total_tokens_needed} tokens needed "
                    f"(messages: {total_message_tokens}, completion: {max_completion_tokens}, "
                    f"safety: {safety_margin}) > {max_context_length} max context length"
                )
                return False
            return True
        except Exception as e:
            logger.warning(f"Error checking judge context window with HF tokenizer: {e}. Falling back to tiktoken.")

    # Fall back to tiktoken.
    try:
        encoding = get_encoding_for_model(model_name)

        total_message_tokens = 0
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")
            content_tokens = len(encoding.encode(content))
            role_tokens = 4 if role == "system" else 3
            total_message_tokens += content_tokens + role_tokens

        total_tokens_needed = total_message_tokens + max_completion_tokens + safety_margin
        if total_tokens_needed > max_context_length:
            logger.warning(
                f"Judge context window would be exceeded: {total_tokens_needed} tokens needed "
                f"(messages: {total_message_tokens}, completion: {max_completion_tokens}, "
                f"safety: {safety_margin}) > {max_context_length} max context length"
            )
            return False

        return True
    except Exception as e:
        logger.warning(f"Error checking judge context window limit: {e}. Proceeding with request.")
        return True  # Default to allowing the request if we can't check


def truncate_messages_to_fit_context(
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    model_name: str,
    max_context_length: int = 8192,
    safety_margin: int = 100,
) -> list[dict[str, str]]:
    """
    Truncate messages to fit within the context window while preserving system messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        max_completion_tokens: Maximum tokens for completion
        model_name: Name of the model (used to determine tokenizer)
        max_context_length: Maximum context length for the model
        safety_margin: Additional safety margin in tokens

    Returns:
        List[Dict[str, str]]: Truncated messages that fit within context window
    """
    # Try cached HuggingFace tokenizer path first.
    tokenizer = _get_hf_tokenizer(model_name)
    if tokenizer is not None:
        try:
            max_context_length = tokenizer.model_max_length if max_context_length is None else max_context_length
            available_tokens = max_context_length - max_completion_tokens - safety_margin

            system_messages = [msg for msg in messages if msg.get("role") == "system"]
            other_messages = [msg for msg in messages if msg.get("role") != "system"]

            system_tokens = 0
            for msg in system_messages:
                content = msg.get("content", "")
                system_tokens += len(tokenizer.encode(content, add_special_tokens=False)) + 4

            remaining_tokens = available_tokens - system_tokens
            if remaining_tokens <= 0:
                logger.warning("System messages alone exceed judge context window. Keeping only system messages.")
                return system_messages

            truncated_messages = system_messages.copy()
            current_tokens = system_tokens

            for msg in other_messages:
                content = msg.get("content", "")
                role = msg.get("role", "")

                content_tokens = len(tokenizer.encode(content, add_special_tokens=False))
                role_tokens = 3
                message_tokens = content_tokens + role_tokens

                if current_tokens + message_tokens <= remaining_tokens:
                    truncated_messages.append(msg)
                    current_tokens += message_tokens
                else:
                    available_for_content = remaining_tokens - current_tokens - role_tokens
                    if available_for_content > 0:
                        content_tokens_encoded = tokenizer.encode(content, add_special_tokens=False)
                        truncated_content = tokenizer.decode(content_tokens_encoded[:available_for_content])
                        truncated_messages.append({"role": role, "content": truncated_content})
                        logger.warning("Truncated message content to fit judge context window")
                    break

            if (
                truncated_messages
                and truncated_messages[-1]["role"] == "user"
                and not truncated_messages[-1]["content"].endswith(
                    'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}'
                )
            ):
                truncated_messages[-1]["content"] = (
                    f'{truncated_messages[-1]["content"]}\nRespond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}'
                )
            return truncated_messages
        except Exception as e:
            logger.warning(f"Error truncating messages with HF tokenizer: {e}. Falling back to tiktoken.")

    # Fall back to tiktoken.
    try:
        encoding = get_encoding_for_model(model_name)

        available_tokens = max_context_length - max_completion_tokens - safety_margin
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        system_tokens = 0
        for msg in system_messages:
            content = msg.get("content", "")
            system_tokens += len(encoding.encode(content)) + 4

        remaining_tokens = available_tokens - system_tokens
        if remaining_tokens <= 0:
            logger.warning("System messages alone exceed judge context window. Keeping only system messages.")
            return system_messages

        truncated_messages = system_messages.copy()
        current_tokens = system_tokens

        for msg in other_messages:
            content = msg.get("content", "")
            role = msg.get("role", "")

            content_tokens = len(encoding.encode(content))
            role_tokens = 3
            message_tokens = content_tokens + role_tokens

            if current_tokens + message_tokens <= remaining_tokens:
                truncated_messages.append(msg)
                current_tokens += message_tokens
            else:
                available_for_content = remaining_tokens - current_tokens - role_tokens
                if available_for_content > 0:
                    truncated_content = encoding.decode(encoding.encode(content)[:available_for_content])
                    truncated_messages.append({"role": role, "content": truncated_content})
                    logger.warning("Truncated message content to fit judge context window")
                break

        if (
            truncated_messages
            and truncated_messages[-1]["role"] == "user"
            and not truncated_messages[-1]["content"].endswith(
                'Respond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}'
            )
        ):
            truncated_messages[-1]["content"] = (
                f'{truncated_messages[-1]["content"]}\nRespond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}'
            )
        return truncated_messages
    except Exception as e:
        logger.warning(f"Error truncating messages: {e}. Returning original messages.")
        return messages


async def safe_acompletion_with_context_check(
    model: str,
    messages: list[dict[str, str]],
    max_completion_tokens: int = 2048,
    max_context_length: int = 8192,
    safety_margin: int = 100,
    **kwargs,
):
    """
    Make an acompletion call with context window checking and automatic truncation.

    Args:
        model: Model name for the completion
        messages: List of message dictionaries
        max_completion_tokens: Maximum tokens for completion
        max_context_length: Maximum context length for the model
        safety_margin: Additional safety margin in tokens
        **kwargs: Additional arguments to pass to acompletion

    Returns:
        The completion response or None if the request cannot be made
    """
    try:
        # Import litellm here to avoid import issues
        from litellm import acompletion

        # Check if the request would exceed context window
        if not check_context_window_limit(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            model_name=model,
            max_context_length=max_context_length,
            safety_margin=safety_margin,
        ):
            # Try to truncate messages to fit
            messages = truncate_messages_to_fit_context(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                model_name=model,
                max_context_length=max_context_length,
                safety_margin=safety_margin,
            )

            # Check again after truncation
            if not check_context_window_limit(
                messages=messages,
                max_completion_tokens=max_completion_tokens,
                model_name=model,
                max_context_length=max_context_length,
                safety_margin=safety_margin,
            ):
                logger.error("Cannot fit request within context window even after truncation.")
                return None

        # Make the acompletion call
        return await acompletion(model=model, messages=messages, max_tokens=max_completion_tokens, **kwargs)

    except Exception as e:
        logger.error(f"Error making safe acompletion call: {e}")
        return None


# Convenience function for quick context checking
def will_exceed_context_window(
    messages: list[dict[str, str]],
    max_completion_tokens: int,
    model_name: str,
    max_context_length: int = 8192,
    safety_margin: int = 100,
) -> bool:
    """
    Quick check to see if a request would exceed the context window.

    Returns:
        bool: True if the request would exceed context window, False otherwise
    """
    return not check_context_window_limit(
        messages, max_completion_tokens, model_name, max_context_length, safety_margin
    )


def truncate_str_for_prompt_template(
    unformatted_str: str,
    prompt_template: str,
    max_completion_tokens: int,
    model_name: str,
    max_context_length: int = 8192,
    safety_margin: int = 100,
    placeholder: str = "{output}",
) -> str:
    """
    Truncates a string to fit into a prompt template within the model's context window.

    Args:
        unformatted_str: The string to be truncated (e.g., a long final answer).
        prompt_template: The prompt template with a placeholder for the string.
        max_completion_tokens: The maximum number of tokens for the completion.
        model_name: The name of the model to determine the tokenizer.
        max_context_length: The maximum context length of the model.
        safety_margin: A safety margin of tokens to leave free.
        placeholder: The placeholder in the template that will be replaced by the string.

    Returns:
        The truncated string.
    """
    try:
        encoding = get_encoding_for_model(model_name)

        # Calculate tokens used by the template itself (without the placeholder content)
        template_without_placeholder = prompt_template.replace(placeholder, "")
        template_tokens = len(encoding.encode(template_without_placeholder))

        # Calculate available tokens for the unformatted string
        available_tokens = max_context_length - template_tokens - max_completion_tokens - safety_margin

        if available_tokens <= 0:
            logger.warning("Prompt template and other params exceed context window. Returning empty string.")
            return ""

        unformatted_str_tokens = encoding.encode(unformatted_str)
        if len(unformatted_str_tokens) > available_tokens:
            truncated_tokens = unformatted_str_tokens[:available_tokens]
            truncated_str = encoding.decode(truncated_tokens)
            logger.warning(
                f"Truncated string from {len(unformatted_str_tokens)} to {len(truncated_tokens)} tokens "
                "to fit into the prompt template."
            )
            return truncated_str
        else:
            return unformatted_str

    except Exception as e:
        logger.warning(f"Error during string truncation: {e}. Returning original string.")
        return unformatted_str
