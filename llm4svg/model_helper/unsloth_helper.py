# -*- coding: utf-8 -*-
# Author: ximing xing
# Copyright (c) 2025, XiMing Xing
# License: MIT License
# Description: Unsloth helper

import torch

# Instruction data template (Alpaca style)
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Chat template placeholder (replace with actual Llama-3 chat template if needed)
CHAT_TEMPLATE_PLACEHOLDER = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{}<|eot_id|>"""


def add_new_tokens(model, tokenizer, new_tokens, logger_instance):
    """Adds new tokens to the tokenizer and resizes model embeddings."""
    logger_instance.info(f"Attempting to add {len(new_tokens)} new tokens.")
    num_added = tokenizer.add_tokens(new_tokens)
    if num_added > 0:
        logger_instance.info(f"Added {num_added} actual new tokens to tokenizer.")
        try:
            model.resize_token_embeddings(len(tokenizer))
            logger_instance.info(f"Resized model token embeddings to: {len(tokenizer)}")
        except Exception as e:
            logger_instance.error(f"Failed to resize token embeddings: {e}", exc_info=True)
    else:
        logger_instance.info("No new tokens were added (they likely existed).")


def format_prompt(instruction, input_text, output_text, eos_token):
    input_part = f"{input_text}" if input_text else ""
    return ALPACA_PROMPT_TEMPLATE.format(instruction, input_part, output_text) + eos_token


def log_gpu_memory_stats(stage, logger_instance, base_memory=0):
    """Logs current GPU memory statistics."""
    if not torch.cuda.is_available():
        logger_instance.info(f"GPU not available. Skipping memory stats for stage: {stage}")
        return base_memory
    try:
        gpu_stats = torch.cuda.get_device_properties(0)
        # Use max_memory_reserved for peak usage during the stage
        reserved_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        # Use memory_allocated for current usage (might be less interesting than peak)
        allocated_memory = round(torch.cuda.memory_allocated() / 1024 / 1024 / 1024, 3)
        total_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger_instance.info(f"--- Memory Stats ({stage}) ---")
        logger_instance.info(f"GPU: {gpu_stats.name}. Total Memory: {total_memory} GB.")
        logger_instance.info(
            f"Current Allocated Memory: {allocated_memory} GB ({round(allocated_memory / total_memory * 100, 2)}%)")
        logger_instance.info(
            f"Peak Reserved Memory ({stage}): {reserved_memory} GB ({round(reserved_memory / total_memory * 100, 2)}%)")
        if base_memory > 0:
            logger_instance.info(f"Delta Peak Reserved Memory ({stage}): {round(reserved_memory - base_memory, 3)} GB")
        logger_instance.info(f"--- End Memory Stats ---")
        return reserved_memory
    except Exception as e:
        logger_instance.warning(f"Could not get GPU memory stats: {e}")
        return base_memory
