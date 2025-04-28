# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: LLM4SVG(Llama3) Training and Inference using unsloth

import logging
from pathlib import Path
import random
import math
from typing import Optional

try:
    from unsloth import UnslothTrainingArguments, UnslothTrainer, FastLanguageModel, apply_chat_template, \
        is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only

except ImportError:
    raise ImportError("Unsloth is not installed or available. This script requires Unsloth.")

from accelerate import Accelerator
import omegaconf
import datasets
import torch
from trl import SFTTrainer
from transformers import GenerationConfig, PreTrainedTokenizerBase, PreTrainedModel
from tqdm.auto import tqdm

from llm4svg.data import syntactic2svg, SVGToken, AttribMapper, ContainerMapper, GradientsMapper, PathCMDMapper, \
    PathMapper, ShapeMapper, NUM_TOKEN, TokenDescMapper, SVGTokenizer
from llm4svg.utils import model_size, write_lines_to_file
from llm4svg.model_helper import ALPACA_PROMPT_TEMPLATE, add_new_tokens, format_prompt, log_gpu_memory_stats

datasets.disable_caching()

# Special tokens: Combine all mappers
combined_mapper = {**PathMapper, **PathCMDMapper, **ShapeMapper, **ContainerMapper,
                   **GradientsMapper, **AttribMapper, **SVGToken}
SVG_TOKEN_LIST = list(combined_mapper.values())


def init_token_embedding_unsloth(
        xcfg: omegaconf.DictConfig,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        logger_instance: logging.Logger
):
    """
    Initializes embeddings for newly added tokens using semantic descriptions.
    Assumes model embeddings have already been resized by add_new_tokens.
    """
    if not (xcfg.use_svg_token and xcfg.get('semantic_init_svg_token', False)):
        logger_instance.info("Skipping semantic token initialization based on config.")
        return model  # Return model unchanged

    logger_instance.info("Attempting semantic initialization for new token embeddings...")

    try:
        # Get the newly added token strings. Handling potential tokenizer differences.
        if hasattr(tokenizer, 'get_added_vocab'):
            added_vocab = tokenizer.get_added_vocab()
            new_tokens = list(added_vocab.keys())
            logger_instance.info(f"Found {len(new_tokens)} added tokens via get_added_vocab.")
        else:
            # Fallback: Identify tokens from SVG_TOKEN_LIST that are actually new
            # This requires knowing the original vocab size before adding tokens
            # Or assuming SVG_TOKEN_LIST contains *only* the potentially new tokens
            # This fallback is less reliable. Let's assume SVG_TOKEN_LIST covers the additions.
            new_tokens = [t for t in SVG_TOKEN_LIST if t in tokenizer.get_vocab()]  # Check they exist now
            logger_instance.warning("Using SVG_TOKEN_LIST to identify new tokens for init (less reliable).")

        if not new_tokens:
            logger_instance.info("No new tokens found to initialize.")
            return model

        embedding_layer = model.get_input_embeddings()
        if embedding_layer is None:
            logger.error("Could not get input embedding layer from the model.")
            return model

        original_embedding_weights = embedding_layer.weight.clone().detach()
        original_vocab_size = original_embedding_weights.shape[0] - len(new_tokens)  # Infer original size

        if original_vocab_size <= 0:
            logger.error(f"Inferred original vocab size ({original_vocab_size}) is invalid.")
            return model

        initialized_count = 0
        skipped_count = 0

        with torch.no_grad():
            for token_str in new_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token_str)

                # Get description from mapper (ensure TokenDescMapper is imported correctly)
                if token_str == NUM_TOKEN:
                    desc = 'number coordinate value'
                else:
                    desc = TokenDescMapper.get(token_str.strip('[]'))

                if not desc:
                    logger_instance.warning(f"No description found for token: '{token_str}'. Skipping semantic init.")
                    skipped_count += 1
                    continue

                # Tokenize description using the *same* tokenizer instance
                try:
                    # Ensure tokenize_ids returns List[int] or similar iterable
                    if isinstance(tokenizer, SVGTokenizer):  # Use specific method if available
                        tokenized_ids = tokenizer.tokenize_ids(desc, padding=False, truncation=True)
                        # Flatten if it returns list of lists for single input
                        if isinstance(tokenized_ids, list) and len(tokenized_ids) > 0 and isinstance(tokenized_ids[0],
                                                                                                     list):
                            tokenized_ids = tokenized_ids[0]
                    else:  # Standard tokenizer
                        tokenized_output = tokenizer(desc,
                                                     add_special_tokens=False)  # Don't add special tokens for init
                        tokenized_ids = tokenized_output['input_ids']

                    # Filter for valid IDs within the *original* vocabulary space
                    valid_original_ids = [id_ for id_ in tokenized_ids if id_ < original_vocab_size]

                except Exception as e:
                    logger_instance.error(f"Error tokenizing description for token '{token_str}': {e}", exc_info=True)
                    skipped_count += 1
                    continue

                if not valid_original_ids:
                    logger_instance.warning(
                        f"Description for token '{token_str}' tokenized to empty list or only new/unknown IDs. Skipping init.")
                    skipped_count += 1
                    continue

                # Calculate mean embedding from valid original tokens
                # Index into the *original* weights, not the potentially modified live weights
                mean_embedding = original_embedding_weights[valid_original_ids, :].mean(dim=0)

                # Assign the calculated embedding to the new token's position in the live layer
                embedding_layer.weight[token_id, :] = mean_embedding
                initialized_count += 1

        logger_instance.info(
            f"Semantic initialization finished. Initialized: {initialized_count}, Skipped: {skipped_count}")

    except Exception as e:
        logger_instance.error(f"Error during semantic token initialization: {e}", exc_info=True)

    return model


def formatting_prompts_func(examples, xcfg, tokenizer):
    """Applies prompt formatting to a batch of examples."""
    instruction_base = xcfg.get("instruction_base", "Generate an SVG illustration from the given description.")
    prompts = examples['text_prompt']
    outputs = examples["svg_text"]
    texts = []
    eos_token = getattr(tokenizer, 'eos_token', '<|eot_id|>')

    for prompt, output in zip(prompts, outputs):
        instruction = f"{instruction_base}"
        input_text = f'SVG illustration of {prompt}'
        text = format_prompt(instruction, input_text, output, eos_token)
        texts.append(text)
    return {"text": texts}


def llama3_sft_by_unsloth(
        cfg: omegaconf.DictConfig,
        project_dir: Path,
        accelerator: Accelerator,
        train_dataset: datasets.Dataset,
        eval_dataset: Optional[datasets.Dataset],
        logger: logging.Logger
):
    xcfg, data_cfg = cfg.x, cfg.data  # Method and dataset config

    # Model and Tokenizer Loading
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/llama-3-8b-bnb-4bit",  # Llama-3 15 trillion tokens model 2x faster!
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
        "unsloth/llama-3-70b-bnb-4bit",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    ]  # More models at https://huggingface.co/unsloth

    if xcfg.model_name not in fourbit_models:
        logger.warning(f"Model {xcfg.model_name} not in known Unsloth 4-bit list. Will attempt 4-bit loading.")

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    logger.info(f"Loading model: {xcfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=xcfg.model_name,
        max_seq_length=xcfg.seq_len,
        dtype=None,
        load_in_4bit=True,
    )

    logger.info(f"Loaded LLM model | size: {model_size(model, 'M'):.1f}M parameters")
    base_mem = log_gpu_memory_stats("Model Loading", logger)

    # Token Handling and Initialization
    original_vocab_size = len(tokenizer)
    if xcfg.use_svg_token:
        # Step 1: Add tokens and resize embeddings
        add_new_tokens(model, tokenizer, SVG_TOKEN_LIST, logger)
        # Step 2: Perform semantic initialization if needed
        if len(tokenizer) > original_vocab_size:
            model = init_token_embedding_unsloth(xcfg, model, tokenizer, logger)
        else:
            logger.info("Vocabulary size did not change, skipping semantic initialization.")

    # Check EOS token AFTER potentially adding tokens
    EOS_TOKEN = getattr(tokenizer, 'eos_token', None)
    if EOS_TOKEN is None:
        logger.error("Tokenizer does not have an EOS token after initialization/token addition.")
        default_eos = '<|eot_id|>'  # Common for Llama-3
        if default_eos in tokenizer.get_vocab():
            tokenizer.eos_token = default_eos
            EOS_TOKEN = default_eos
            logger.warning(f"Setting EOS token to '{default_eos}'.")
        else:
            logger.warning(f"Attempting to add missing EOS token: {default_eos}")
            num_added = tokenizer.add_special_tokens({'eos_token': default_eos})
            if num_added > 0:
                model.resize_token_embeddings(len(tokenizer))
                logger.info("Added EOS token and resized embeddings.")
                EOS_TOKEN = default_eos
            else:
                raise ValueError("EOS_TOKEN is None and could not be added/set. Cannot proceed.")

    # Dataset Preparation
    logger.info("Formatting dataset prompts...")
    format_func = lambda examples: formatting_prompts_func(examples, xcfg, tokenizer)
    train_dataset = train_dataset.map(
        format_func,
        batched=True,
        num_proc=data_cfg.num_workers,
        remove_columns=[col for col in train_dataset.column_names if col not in ['text']],
        desc="Formatting train prompts"
    )
    logger.info(f"Train dataset columns after formatting: {train_dataset.column_names}")

    # Log a few random samples from the training set (main process only)
    if accelerator.is_main_process:
        num_samples_to_log = min(2, len(train_dataset))
        if num_samples_to_log > 0:
            indices = random.sample(range(len(train_dataset)), num_samples_to_log)
            for index in indices:
                logger.info(f"--- Train Sample {index} ---")
                logger.info(train_dataset[index]['text'])
                logger.info(f"--- End Sample ---")
        else:
            logger.info("Train dataset is empty. Cannot log samples.")

    # PEFT Configuration
    target_modules = xcfg.get("lora_target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    # Add optional modules based on config flags
    if xcfg.ft_lm_head and "lm_head" not in target_modules:
        target_modules.append("lm_head")
        logger.info("Including 'lm_head' in LoRA target modules.")
    if xcfg.ft_embed_tokens and "embed_tokens" not in target_modules:
        target_modules.append("embed_tokens")
        logger.info("Including 'embed_tokens' in LoRA target modules.")

    logger.info(f"Applying PEFT (LoRA) to the model with target modules: {target_modules}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=xcfg.lora_rank,
        target_modules=target_modules,
        lora_alpha=xcfg.lora_alpha,
        lora_dropout=xcfg.lora_dropout,
        bias=xcfg.peft_bias,
        use_gradient_checkpointing=xcfg.use_gradient_checkpointing,  # "unsloth" uses less VRAM
        random_state=cfg.seed,
        max_seq_length=xcfg.seq_len,
        use_rslora=xcfg.get('use_rslora', False),
        loftq_config=None,
    )
    logger.info("PEFT applied. Trainable parameters:")
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Trainable PEFT parameters: {trainable_params}")

    # Training Arguments
    resume_from_checkpoint = xcfg.get('resume_from_checkpoint', False)
    resume_dir = xcfg.get('resume_dir', None)
    checkpoint_path = Path(resume_dir) if resume_dir else None
    resume_arg = checkpoint_path.as_posix() if resume_from_checkpoint and checkpoint_path and checkpoint_path.is_dir() else None
    if resume_from_checkpoint and not resume_arg:
        logger.warning(f"Resume specified but path '{resume_dir}' not found or not a directory. Training from scratch.")

    # Ensure output_dir exists
    output_dir = project_dir / "training_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = UnslothTrainingArguments(
        per_device_train_batch_size=xcfg.train_batch_size,
        gradient_accumulation_steps=xcfg.gradient_accumulation_steps,
        warmup_steps=xcfg.warmup_steps,
        num_train_epochs=xcfg.get('num_train_epochs', 3),
        learning_rate=xcfg.lr,
        # Use embedding_learning_rate if ft_embed_tokens is True and rate is specified
        embedding_learning_rate=xcfg.embedding_learning_rate \
            if xcfg.ft_embed_tokens and xcfg.embedding_learning_rate else None,
        lr_scheduler_type=xcfg.lr_scheduler,
        weight_decay=xcfg.weight_decay,
        max_grad_norm=xcfg.grad_max_norm,
        logging_steps=xcfg.log_steps,
        optim="adamw_8bit",
        save_strategy="steps" if xcfg.save_steps > 0 else "epoch",
        save_steps=xcfg.save_steps if xcfg.save_steps > 0 else 0,
        save_total_limit=xcfg.get('save_total_limit', 3),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        output_dir=output_dir.as_posix(),
        dataloader_num_workers=data_cfg.num_workers,
        seed=cfg.seed,
        report_to=xcfg.report_to if xcfg.with_tracking else "none",
        gradient_checkpointing=xcfg.use_gradient_checkpointing,
        ddp_find_unused_parameters=xcfg.get('ddp_find_unused_parameters', None),
    )

    # Trainer Initialization
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if xcfg.get("do_eval", False) else None,
        dataset_text_field="text",
        max_seq_length=xcfg.seq_len,
        args=training_args,
        packing=xcfg.get('packing', True),
    )

    # Train on completions / responses only (Do not train on inputs)
    if xcfg.train_on_responses_only:
        logger.info("Configuring trainer to train on responses only (Unsloth).")
        try:
            trainer = train_on_responses_only(trainer)
        except Exception as e:
            logger.error(f"Failed to apply train_on_responses_only. Check chat template compatibility. Error: {e}")

    # Training
    logger.info("Starting training...")
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_arg)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise e

    # Post-Training
    logger.info("Training finished.")
    log_gpu_memory_stats("Training", logger, base_memory=base_mem)

    # Log training stats
    try:
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        logger.info(f"Training Metrics: {metrics}")
    except Exception as e:
        logger.warning(f"Could not log/save training metrics: {e}")

    # Save final model, adapter, tokenizer, and state
    if accelerator.is_main_process:
        final_model_dir = project_dir / "final_model"
        logger.info(f"Saving final model adapter and tokenizer to {final_model_dir}...")
        try:
            trainer.save_model(final_model_dir.as_posix())
            tokenizer.save_pretrained(final_model_dir.as_posix())
            trainer.save_state()
            logger.info("Final model adapter, tokenizer, and trainer state saved.")

            if xcfg.get('save_merged_model', False):
                logger.info("Merging adapter weights and saving full model...")
                merged_dir = project_dir / "final_merged_model"
                merged_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Use the model from the trainer
                    peft_model = trainer.model
                    # Merge and unload - returns the base model with weights merged
                    merged_model = peft_model.merge_and_unload()
                    merged_model.save_pretrained(merged_dir.as_posix(), safe_serialization=True)
                    tokenizer.save_pretrained(merged_dir.as_posix())
                    logger.info(f"Full merged model saved to {merged_dir}")
                except Exception as merge_err:
                    logger.error(f"Failed to merge and save model: {merge_err}", exc_info=True)

        except Exception as e:
            logger.error(f"Error saving final model/adapter: {e}", exc_info=True)
    # Ensure all processes wait before proceeding (e.g., to inference)
    accelerator.wait_for_everyone()

    # Inference
    if xcfg.get('run_inference_after_train', True):
        logger.info("Running inference on evaluation dataset...")
        inference_model = trainer.model

        FastLanguageModel.for_inference(inference_model)

        try:
            inference(
                cfg=cfg,
                accelerator=accelerator,
                model=inference_model,
                tokenizer=tokenizer,
                eval_dataset=eval_dataset,
                project_dir=project_dir,
                logger_instance=logger
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}", exc_info=True)
    else:
        logger.info("Skipping inference after training as per configuration.")

    logger.info("--- Unsloth Llama-3 SFT Script Finished ---")


@torch.no_grad()
def inference(
        cfg: omegaconf.DictConfig,
        accelerator: Accelerator,
        model: torch.nn.Module,  # Expects the potentially PEFT model
        tokenizer: PreTrainedTokenizerBase,
        eval_dataset: datasets.Dataset,
        project_dir: Path,
        logger_instance: logging.Logger  # Pass logger
):
    """
    Performs inference using the trained model.

    Args:
        cfg: OmegaConf configuration object.
        accelerator: Instance of Accelerate's Accelerator.
        model: The trained model instance (potentially PEFT).
        tokenizer: The tokenizer instance.
        eval_dataset: Evaluation dataset.
        project_dir: Path to the main project/output directory.
        logger_instance: Configured logger instance.
    """
    global logger
    logger = logger_instance

    if eval_dataset is None or len(eval_dataset) == 0:
        logger.warning("Evaluation dataset is empty or None. Skipping inference.")
        return

    xcfg = cfg.x
    gen_cfg = xcfg.get('generation', omegaconf.OmegaConf.create({}))

    device = accelerator.device
    model.eval()

    # Prepare Output Directory
    sample_dir = project_dir / "samples"
    sample_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving generated samples to: {sample_dir}")

    # Prepare Generation Configuration
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
        logger.warning(f"Tokenizer pad_token_id is None. Using eos_token_id ({pad_token_id}) for generation.")

    generation_config = GenerationConfig(
        max_new_tokens=gen_cfg.get('max_new_tokens', 512),
        do_sample=gen_cfg.get('do_sample', True),
        temperature=gen_cfg.get('temperature', 0.6),
        top_p=gen_cfg.get('top_p', 0.9),
        top_k=gen_cfg.get('top_k', 50),
        pad_token_id=pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    logger.info(f"Using Generation Config: {generation_config}")

    # Prepare Input Prompts
    logger.info("Preparing evaluation prompts...")
    eval_prompts = []
    instruction_base = xcfg.get("instruction_base", "Generate an SVG illustration from the given description.")

    # Ensure 'text_prompt' exists in eval_dataset
    if 'text_prompt' not in eval_dataset.column_names:
        logger.error("Evaluation dataset must contain 'text_prompt' column for inference.")
        return

    for example in tqdm(eval_dataset, desc="Preparing Prompts", disable=not accelerator.is_local_main_process):
        prompt_text = example['text_prompt']
        instruction = f"{instruction_base}"
        input_context = f'SVG illustration of {prompt_text}'

        # Format the prompt for inference (only the input part)
        formatted_prompt = ALPACA_PROMPT_TEMPLATE.format(instruction, input_context, '').split("### Response:")[
                               0] + "### Response:"

        eval_prompts.append(formatted_prompt)

    # Batch Generation
    eval_batch_size = gen_cfg.get('eval_batch_size', xcfg.train_batch_size)
    eval_batch_size = max(1, eval_batch_size)

    num_batches = math.ceil(len(eval_prompts) / eval_batch_size)
    logger.info(f"Generating {len(eval_prompts)} samples in {num_batches} batches (size {eval_batch_size}).")

    all_results_text = []  # Store raw text outputs
    all_results_svg = []  # Store converted SVG outputs

    for i, batch_idx in enumerate(tqdm(range(0, len(eval_prompts), eval_batch_size), desc="Generating Batches",
                                       disable=not accelerator.is_local_main_process)):
        batch_prompts = eval_prompts[batch_idx: batch_idx + eval_batch_size]
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=xcfg.seq_len
        ).to(device)

        model_to_generate = model
        try:
            batch_outputs = model_to_generate.generate(
                input_ids=batch_inputs.input_ids,
                attention_mask=batch_inputs.attention_mask,
                generation_config=generation_config
            )
        except Exception as gen_e:
            logger.error(f"Error during model.generate for batch {i}: {gen_e}", exc_info=True)
            continue

        # Decode outputs
        input_lengths = batch_inputs.input_ids.shape[1]
        generated_ids = batch_outputs[:, input_lengths:]
        batch_decoded_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for idx, decoded_text in enumerate(batch_decoded_text):
            original_prompt_index = batch_idx + idx
            if original_prompt_index >= len(eval_dataset): continue
            original_prompt_text = eval_dataset[original_prompt_index]['text_prompt']

            result_str = f"--- Sample {original_prompt_index + 1} ---\n"
            result_str += f"Prompt: {original_prompt_text}\n"
            result_str += f"Generated Text:\n{decoded_text}\n"
            all_results_text.append(result_str)

            # Attempt to convert to SVG
            try:
                # Ensure decoded_text is not empty before conversion
                if decoded_text.strip():
                    svg_code = syntactic2svg(decoded_text)
                    svg_filename = sample_dir / f"sample_{original_prompt_index + 1}.svg"
                    write_lines_to_file(svg_filename.as_posix(), [svg_code])
                    all_results_svg.append(f"--- Sample {original_prompt_index + 1} ---")
                    all_results_svg.append(f"Prompt: {original_prompt_text}")
                    all_results_svg.append(f"Generated SVG (saved to {svg_filename.name}):\n{svg_code}\n")
                else:
                    logger.warning(
                        f"Generated text for sample {original_prompt_index + 1} is empty. Skipping SVG conversion.")
                    all_results_svg.append(f"--- Sample {original_prompt_index + 1} ---")
                    all_results_svg.append(f"Prompt: {original_prompt_text}")
                    all_results_svg.append(f"Generated Text was empty. SVG conversion skipped.\n")

            except Exception as e:
                logger.warning(f"Failed to convert generated text to SVG for sample {original_prompt_index + 1}: {e}")
                all_results_svg.append(f"--- Sample {original_prompt_index + 1} ---")
                all_results_svg.append(f"Prompt: {original_prompt_text}")
                all_results_svg.append(f"Generated Text (SVG conversion failed: {e}):\n{decoded_text}\n")

    if accelerator.is_main_process:
        text_output_file = sample_dir / "all_generated_text.txt"
        svg_output_file = sample_dir / "all_generated_svg_log.txt"

        try:
            write_lines_to_file(text_output_file.as_posix(), all_results_text)
            logger.info(f"Saved all generated text outputs to {text_output_file}")
        except Exception as e:
            logger.error(f"Failed to save aggregated text results: {e}")

        try:
            write_lines_to_file(svg_output_file.as_posix(), all_results_svg)
            logger.info(f"Saved all generated SVG results (with conversion status) to {svg_output_file}")
        except Exception as e:
            logger.error(f"Failed to save aggregated SVG results: {e}")

    logger.info("--- Inference Finished ---")
