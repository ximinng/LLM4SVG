# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: LLM4SVG(GPT-2) Training Script

import copy
import logging
import random
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import omegaconf
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import Dataset as HFDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    get_scheduler,
)

from llm4svg.data import NUM_TOKEN, SVGTokenizer, TokenDescMapper
from llm4svg.svglib import save_svg_text
from llm4svg.utils import (
    EMA,
    compute_train_schedule,
    create_adam_optimizer,
    model_size,
    path_exists,
)


def init_token_embedding(
    xcfg: omegaconf.DictConfig,
    model: GPT2LMHeadModel,
    svg_tokenizer: SVGTokenizer,
    logger: logging.Logger,
):
    """Token Embedding Initialization"""
    update_vocab_size = len(svg_tokenizer)
    original_embedding_layer = model.get_input_embeddings()
    origin_vocab_size = original_embedding_layer.weight.shape[0]
    if update_vocab_size > origin_vocab_size:
        logger.info(
            f"Resizing token embeddings from {origin_vocab_size} to {update_vocab_size}"
        )
        model.resize_token_embeddings(update_vocab_size)

    # Reinitialize new token embeddings if specified
    if xcfg.semantic_init_svg_token:
        new_tokens = svg_tokenizer.get_new_tokens()
        logger.info(
            f"Initializing embeddings for {len(new_tokens)} new tokens using descriptions."
        )

        with torch.no_grad():
            for i, token in enumerate(reversed(new_tokens), start=1):
                if "[" or "]" in token:
                    token = token.replace("[", "").replace("]", "")
                if token == NUM_TOKEN:
                    desc = "number, the value of the coordinate"
                    logger.info(f"Using description for NUM_TOKEN: '{desc}'")
                else:
                    if token not in TokenDescMapper.keys():
                        logger.info(
                            f"[Warning] No description found in TokenDescMapper for new token: '{token}'."
                            f" Skipping initialization."
                        )
                        continue
                    else:
                        desc = TokenDescMapper[token]

                # Tokenize description
                tokenized = svg_tokenizer.tokenize(desc)
                tokenized_ids = svg_tokenizer.tokens2ids(tokenized)

                # Filter out unknown tokens if any resulted from tokenization
                valid_ids = [
                    id_ for id_ in tokenized_ids if id_ < origin_vocab_size
                ]  # Only use original vocab IDs
                if not valid_ids:
                    logger.info(
                        f"[Warning] Description for token '{token}' tokenized to empty or all-new/unknown IDs. "
                        f"Cannot initialize embedding from description."
                    )
                    continue

                # Get embeddings of description tokens (from original embeddings)
                # Ensure we only index within the bounds of the *original* embedding matrix size
                description_embeddings = original_embedding_layer.weight[
                    valid_ids, :
                ].mean(axis=0)

                # Assign the calculated embedding to the new token's position
                model.transformer.wte.weight[-i, :] = (
                    description_embeddings.clone().detach()
                )
    # model.tokenizer = svg_tokenizer
    return model


@torch.no_grad()
def generate_svg(
    prompt: str,
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast | SVGTokenizer,
    device: torch.device,
    pad_token_id: int,
    max_length: int = 100,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    post_processing: bool = True,
) -> List[str]:
    """Generates SVG text sequence from a prompt using the provided model."""

    model.eval()  # Ensure model is in eval mode

    # Tokenize prompt
    eval_input = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True)

    input_ids = eval_input["input_ids"].to(device)
    attention_mask = eval_input.get("attention_mask", torch.ones_like(input_ids)).to(
        device
    )

    # Find the token ID for '</svg>'
    # Assuming '</svg>' is a single token in the SVGTokenizer's vocabulary
    svg_end_token = "</svg>"
    svg_end_token_id = tokenizer.convert_tokens_to_ids(svg_end_token)

    if svg_end_token_id is None or svg_end_token_id == tokenizer.unk_token_id:
        # If '</svg>' is not a recognized token
        print(
            f"[Warning] Token '{svg_end_token}' not found in tokenizer vocabulary. "
            f"Generation will not stop explicitly on this token. "
            f"UNK ID: {tokenizer.unk_token_id}, SVG_END ID: {svg_end_token_id}"
        )
        # Use the default EOS token for the model if available
        stop_token_id = (
            model.config.eos_token_id
            if model.config.eos_token_id is not None
            else pad_token_id
        )
    else:
        # Use the found ID as the stopping criterion
        stop_token_id = svg_end_token_id
        print(
            f"Set stop token ID for generation to '{svg_end_token}' (ID: {stop_token_id})"
        )

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        max_length=max_length,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=stop_token_id,
    )

    # Decode generated sequences
    generated_texts = [
        tokenizer.decode(
            o, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for o in output
    ]

    if post_processing:
        # Trim the text after the first occurrence of '</svg>' if it wasn't the actual stop token
        trimmed_texts = []
        for text in generated_texts:
            svg_end_index = text.find(svg_end_token)
            if svg_end_index != -1:
                # Include the '</svg>' token itself
                trimmed_texts.append(text[: svg_end_index + len(svg_end_token)])
            else:
                trimmed_texts.append(
                    text
                )  # No '</svg>' found, return the full generated text

        return trimmed_texts

    return generated_texts


def collate_fn(
    batch: List[Dict[str, List[int]]], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """Pads sequences dynamically within a batch."""
    input_ids = [torch.LongTensor(example["input_ids"]) for example in batch]
    labels = [torch.LongTensor(example["labels"]) for example in batch]
    # Pad sequences to the maxlength in this batch
    input_ids_padded = pad_sequence(
        input_ids, batch_first=True, padding_value=pad_token_id
    )
    labels_padded = pad_sequence(
        labels, batch_first=True, padding_value=-100
    )  # Use -100 for label padding

    collated_batch = {"input_ids": input_ids_padded, "labels": labels_padded}
    # Handle attention mask if present
    if (
        "attention_mask" in batch[0]
    ):  # Check if attention_mask exists in the tokenized output
        attention_masks = [
            torch.LongTensor(example["attention_mask"]) for example in batch
        ]
        attention_mask_padded = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )  # Pad attention mask with 0
        collated_batch["attention_mask"] = attention_mask_padded

    return collated_batch


def llm4svg_gpt2_sft(
    cfg: omegaconf.DictConfig,
    project_dir: Path,
    accelerator: Accelerator,
    dataset: HFDataset,
    logger: logging.Logger,
):
    xcfg, data_cfg = cfg.x, cfg.data
    device = accelerator.device

    logger.info("--- Starting LLM4SVG Training ---")
    logger.info(f"Project Directory: {project_dir}")
    logger.info(f"Configuration: {omegaconf.OmegaConf.to_yaml(cfg)}")  # Log config

    # Load Tokenizer
    logger.info("Loading tokenizer...")
    if xcfg.use_svg_token:
        tokenizer = SVGTokenizer(
            xcfg, print_fn=logger.info, local_files_only=xcfg.local_file
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            xcfg.model_name, local_files_only=xcfg.local_file
        )

    # Ensure PAD token is set (GPT-2 often doesn't have one by default)
    # Line crashes for SVGTokenzier, in this case, the pad_token is accessed from the underlying tokenizer
    if tokenizer.tokenizer.pad_token is None:
        tokenizer.tokenizer.pad_token = tokenizer.tokenizer.eos_token
        logger.info(
            f"Set PAD token to EOS token: {tokenizer.tokenizer.eos_token} (ID: {tokenizer.eos_token_id})"
        )
    pad_token_id = tokenizer.pad_token_id

    if xcfg.save_tokenizer and accelerator.is_main_process:
        tokenizer_save_dir = project_dir / "tokenizer"
        tokenizer_save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_save_dir.as_posix())
        logger.info(f"Tokenizer saved to: {tokenizer_save_dir}")

    # Load Model Configuration
    logger.info("Loading model config...")
    config = AutoConfig.from_pretrained(
        xcfg.model_name, local_files_only=xcfg.local_file
    )
    # Update config BEFORE loading model
    original_n_positions = config.n_positions
    if original_n_positions != xcfg.seq_len:
        logger.info(
            f"Updating model max sequence length (n_positions) from {original_n_positions} to {xcfg.seq_len}"
        )
        config.n_positions = xcfg.seq_len
    # Add pad_token_id to config if not present
    config.pad_token_id = pad_token_id

    logger.info(f"Model Name: {xcfg.model_name}")
    logger.info(f"Effective Model Config:\n{config}")

    # Load Model
    logger.info("Loading model...")
    if xcfg.train_scratch:
        logger.info("Training model FROM SCRATCH")
        model = AutoModelForCausalLM.from_config(config)
    else:
        logger.info(f"Loading pre-trained model: {xcfg.model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            xcfg.model_name,
            config=config,
            ignore_mismatched_sizes=True,
            local_files_only=xcfg.local_file,
        )

    logger.info(f"Model Size: {model_size(model, unit='M'):.2f}M parameters")

    # Initialize token embeddings
    model = init_token_embedding(xcfg, model, tokenizer, logger)
    if model.config.vocab_size != len(tokenizer):
        logger.info(f"[Info] Updating model config vocab_size to {len(tokenizer)}")
        model.config.vocab_size = len(tokenizer)

    # EMA Configuration
    # Add these to your OmegaConf config file (e.g., in the 'x' section)
    use_ema = xcfg.get("use_ema", False)  # Default EMA to False
    ema_beta = xcfg.get("ema_beta", 0.9999)
    ema_update_after_step = xcfg.get("ema_update_after_step", 100)
    ema_update_every = xcfg.get("ema_update_every", 10)
    ema_eval_use_ema_weights = xcfg.get("ema_eval_use_ema_weights", True)

    logger.info(f"Use EMA: {use_ema}")
    if use_ema:
        logger.info(f"  EMA Beta: {ema_beta}")
        logger.info(f"  EMA Update After Step: {ema_update_after_step}")
        logger.info(f"  EMA Update Every: {ema_update_every}")
        logger.info(f"  Evaluate with EMA weights: {ema_eval_use_ema_weights}")

    # Initialize EMA
    ema_instance: Optional[EMA] = None
    if use_ema:
        logger.info("Initializing EMA...")
        ema_instance = EMA(
            model,
            beta=ema_beta,
            update_after_step=ema_update_after_step,
            update_every=ema_update_every,
        )
        # Register the EMA instance for automatic checkpoint saving/loading
        # Ensure ema_instance itself is treated correctly by accelerator's state handling
        if isinstance(ema_instance, nn.Module):  # EMA inherits from nn.Module
            accelerator.register_for_checkpointing(ema_instance)
            logger.info("Registered EMA instance for checkpointing.")
        else:
            logger.info(
                "[Warning] EMA instance is not an nn.Module, manual checkpointing might be needed."
            )

    def preprocess_data(example):
        prompt_part = f"{example['text_prompt']}. output:"
        output_part = str(example["svg_text"])
        return {"input_text": f"{prompt_part}{output_part}", "prompt_part": prompt_part}

    logger.info(f"Processing dataset. Original columns: {dataset.column_names}")
    # Add prompt_part column for use in tokenize_function
    dataset = dataset.map(
        preprocess_data, num_proc=data_cfg.num_workers, desc="Organizing input data"
    )
    logger.info(f"Dataset columns after preprocess: {dataset.column_names}")

    # Ignoring prompt loss
    ignore_prompt_loss = xcfg.get("ignore_prompt_loss", True)
    logger.info(
        f"Configuring loss calculation: ignore_prompt_loss = {ignore_prompt_loss}"
    )

    def tokenize_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """
        Tokenizes batches and prepares labels, masking prompt tokens if ignore_prompt_loss is True.
        """
        num_examples = len(examples["input_text"])
        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        # Use tokenizer on the batch of full texts
        full_tokenized = tokenizer(
            examples["input_text"],
            max_length=xcfg.seq_len,
            padding=False,
            truncation=True,
        )

        if ignore_prompt_loss:
            prompt_parts = examples["prompt_part"]
            prompt_tokenized = tokenizer(
                prompt_parts, max_length=xcfg.seq_len, padding=False, truncation=True
            )

        for i in range(num_examples):
            current_input_ids = full_tokenized["input_ids"][i]
            current_labels = current_input_ids.copy()  # Start with labels = input_ids

            if ignore_prompt_loss:
                # Get the length of the tokenized prompt for this specific example
                prompt_tokens_length = len(prompt_tokenized["input_ids"][i])
                # Safety check: make sure prompt length isn't longer than total length
                mask_len = min(prompt_tokens_length, len(current_labels))
                # Set labels for prompt tokens to -100
                current_labels[:mask_len] = [-100] * mask_len

            input_ids_batch.append(current_input_ids)
            labels_batch.append(current_labels)
            # Add attention mask if tokenizer provides it
            if "attention_mask" in full_tokenized:
                attention_mask_batch.append(full_tokenized["attention_mask"][i])

        batch = {"input_ids": input_ids_batch, "labels": labels_batch}
        if attention_mask_batch:  # Check if we collected any attention masks
            batch["attention_mask"] = attention_mask_batch
        return batch

    logger.info("Tokenizing dataset and preparing labels...")
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=data_cfg.num_workers,
        remove_columns=list(dataset.column_names),
        desc="Running tokenizer and preparing labels",
    )
    logger.info(f"Tokenized dataset features: {tokenized_datasets.features}")
    logger.info(f"Tokenized dataset columns: {tokenized_datasets.column_names}\n")

    # Filter samples with a length exceeding `seq_len`
    original_count = len(tokenized_datasets)
    logger.info(f"Tokenized dataset size: {original_count} examples.")

    def filter_long_sequences(example):
        return (
            len(example["input_ids"]) <= xcfg.seq_len
        )  # Keep if length is within limit

    logger.info(f"Filtering out sequences longer than {xcfg.seq_len} tokens...")
    filtered_datasets = tokenized_datasets.filter(
        filter_long_sequences,
        num_proc=data_cfg.num_workers,
        desc="Filtering long sequences",
    )
    filtered_count = len(filtered_datasets)
    removed_count = original_count - filtered_count

    logger.info(
        f"Removed {removed_count} examples exceeding max sequence length ({xcfg.seq_len})."
    )
    logger.info(f"Final dataset size after filtering: {filtered_count} examples.")

    # Check if dataset is empty after filtering
    if filtered_count == 0:
        logger.error("Dataset is empty after filtering! Check max_seq_len or data.")
        raise ValueError("Dataset empty after filtering.")

    # Log a few random samples
    if accelerator.is_local_main_process:
        for index in random.sample(range(len(tokenized_datasets)), 2):
            logger.info(f"--- Sample {index} ---")
            decoded_text = tokenizer.decode(tokenized_datasets[index]["input_ids"])
            logger.info(f"Input Text (decoded): {decoded_text}")  # Print truncated
            logger.info(
                f"Input IDs: {tokenized_datasets[index]['input_ids']}"
            )  # Print truncated
            logger.info(
                f"Labels: {tokenized_datasets[index]['labels']}"
            )  # Print truncated
            logger.info(f"--- End Sample {index} ---\n")

    # DataLoaders Creation
    logger.info("Creating DataLoader...")
    train_dataloader = DataLoader(
        tokenized_datasets,
        shuffle=True,
        batch_size=xcfg.train_batch_size,
        collate_fn=partial(collate_fn, pad_token_id=pad_token_id),
        num_workers=data_cfg.num_workers,
    )

    # Optimizer and Scheduler
    logger.info("Creating optimizer and scheduler...")
    optimizer = create_adam_optimizer(
        model, lr=xcfg.lr, weight_decay=xcfg.weight_decay, split_decay_params=True
    )
    # Compute training schedule details
    max_train_steps, num_train_epochs, completed_steps, starting_epoch = (
        compute_train_schedule(
            train_dataloader,
            xcfg.num_train_epochs,
            xcfg.gradient_accumulation_steps,
            xcfg.resume_step,
        )
    )
    # Learning Rate Scheduler
    lr_scheduler = get_scheduler(
        name=xcfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=xcfg.warmup_steps
        * accelerator.num_processes,  # Scale warmup steps
        num_training_steps=max_train_steps,
    )
    logger.info(
        f"Scheduler: {xcfg.lr_scheduler}, Warmup Steps: {xcfg.warmup_steps}, Total Steps: {max_train_steps}"
    )

    logger.info("Preparing model, optimizer, dataloader, scheduler with Accelerator...")
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # Load Checkpoint
    if xcfg.resume_from_checkpoint and path_exists(xcfg.resume_from_checkpoint):
        checkpoint_path = Path(xcfg.resume_from_checkpoint)
        if checkpoint_path.is_dir():
            logger.info(f"Resuming from checkpoint directory: {checkpoint_path}")
            accelerator.load_state(checkpoint_path.as_posix())
            logger.info(
                f"Resumed state. Current step: {completed_steps}, Current epoch: {starting_epoch}"
            )
            # EMA state should be loaded automatically if registered
            if use_ema and ema_instance:
                logger.info(
                    f"EMA state loaded. EMA current step: {ema_instance.num_updates}"
                )
        else:
            logger.info(
                f"[Warning] Checkpoint path specified but not found or not a directory: {checkpoint_path}. Starting from scratch."
            )
            # Reset schedule vars if checkpoint load failed but was expected
            completed_steps, starting_epoch = 0, 0

    # Initialize Trackers
    if xcfg.with_tracking and accelerator.is_main_process:
        tracker_config = omegaconf.OmegaConf.to_container(
            xcfg, resolve=True, throw_on_missing=True
        )
        accelerator.init_trackers(project_name=xcfg.project_name, config=tracker_config)
        logger.info("Initialized trackers.")

    # Training Start
    total_batch_size = (
        xcfg.train_batch_size
        * accelerator.num_processes
        * xcfg.gradient_accumulation_steps
    )

    logger.info("\n***** Running training *****")
    logger.info(f"  Num examples = {len(tokenized_datasets)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Start Epoch = {starting_epoch}")
    logger.info(f"  Instantaneous batch size per device = {xcfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {xcfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Start Step = {completed_steps}")
    logger.info(f"  Mixed precision = {accelerator.mixed_precision}")
    accelerator.wait_for_everyone()

    # Progress bar setup
    progress_bar = tqdm(
        range(max_train_steps),
        initial=completed_steps,
        disable=not accelerator.is_local_main_process,
        desc="Training Progress",
    )

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss_accumulated = 0.0  # reset loss accumulator for the epoch

        for step, batch in enumerate(train_dataloader):
            # Gradient Accumulation Context
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                # Accumulate loss from each micro-batch
                total_loss_accumulated += loss.item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if xcfg.grad_max_norm > 0:
                        accelerator.clip_grad_norm_(
                            model.parameters(), float(xcfg.grad_max_norm)
                        )

                # Optimizer Step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # EMA Update
                if accelerator.sync_gradients and use_ema and ema_instance:
                    ema_instance.update()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                avg_loss = total_loss_accumulated / xcfg.gradient_accumulation_steps
                current_lr = optimizer.param_groups[0]["lr"]

                progress_bar.set_description(
                    f"Epoch {epoch + 1}/{num_train_epochs} | Step {step + 1}/{len(train_dataloader)} | "
                    f"LR: {current_lr:.1e} | Loss: {avg_loss:.4f}"
                )

                if xcfg.with_tracking:
                    accelerator.log(
                        {
                            "loss": avg_loss,
                            "lr": current_lr,
                            "epoch": epoch,
                            "step": completed_steps,
                        },
                        step=completed_steps,
                    )

                # Add Periodic Loss Logging to File
                log_interval = cfg.logging.log_interval
                is_log_step = (
                    completed_steps == 1
                    or (log_interval > 0 and completed_steps % log_interval == 0)
                    or completed_steps == max_train_steps
                )

                if is_log_step:
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step: {completed_steps}/{max_train_steps} | "
                            f"Epoch: {epoch + 1}/{num_train_epochs} | "
                            f"Avg Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.1e}"
                        )

                total_loss_accumulated = 0.0  # reset accumulator for next set of steps

                # Evaluation and Checkpointing
                if completed_steps > 0 and completed_steps % xcfg.eval_steps == 0:
                    logger.info(
                        f"\nStep {completed_steps}: Running evaluation and saving checkpoint..."
                    )
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        save_dir = project_dir / f"step_{completed_steps}"
                        state_dir = save_dir / "states"
                        sample_dir = save_dir / "samples"
                        state_dir.mkdir(parents=True, exist_ok=True)
                        sample_dir.mkdir(parents=True, exist_ok=True)

                        accelerator.save_state(state_dir.as_posix())
                        logger.info(f"Saved accelerator state to {state_dir}")

                        # Generate Samples
                        unwrapped_model = accelerator.unwrap_model(model)

                        # Prepare model for evaluation
                        eval_model = (
                            unwrapped_model  # default to the unwrapped trained model
                        )
                        original_state_dict = (
                            None  # to store original weights if swapping
                        )

                        if use_ema and ema_instance and ema_eval_use_ema_weights:
                            logger.info("Using EMA weights for evaluation.")
                            original_state_dict = copy.deepcopy(
                                unwrapped_model.state_dict()
                            )
                            # Access the EMA model's parameters (ema_model.ema_model)
                            try:
                                ema_weights = ema_instance.ema_model.state_dict()
                                unwrapped_model.load_state_dict(ema_weights)
                                logger.info(
                                    "Swapped model weights with EMA weights for generation."
                                )
                                eval_model = unwrapped_model  # Use the model with EMA weights loaded
                            except Exception as e:
                                logger.error(
                                    f"[Error] Failed to load EMA weights for evaluation: {e}. Using standard model weights."
                                )
                                # Ensure original weights are restored if swap started but failed
                                if original_state_dict:
                                    unwrapped_model.load_state_dict(original_state_dict)
                                eval_model = unwrapped_model  # Fallback to original

                        # Consider using a fixed validation set for consistent evaluation
                        logger.info("Generating evaluation samples...")
                        for i in range(xcfg.eval_sample):
                            # Get eval prompt
                            random_idx = random.randint(0, len(dataset) - 1)
                            random_example = dataset[random_idx]
                            prompt = f"{random_example['text_prompt']}. output:"

                            # Get generation parameters from config or use defaults
                            gen_max_length = xcfg.get(
                                "gen_max_length", xcfg.seq_len
                            )  # Default to context window size
                            gen_temp = xcfg.get("gen_temp", 1.0)
                            gen_top_k = xcfg.get("gen_top_k", 50)
                            gen_top_p = xcfg.get("gen_top_p", 0.95)
                            gen_do_sample = xcfg.get("gen_do_sample", True)

                            generated_texts = generate_svg(
                                prompt=prompt,
                                model=eval_model,
                                tokenizer=tokenizer,
                                device=device,
                                pad_token_id=pad_token_id,
                                max_length=gen_max_length,
                                temperature=gen_temp,
                                top_k=gen_top_k,
                                top_p=gen_top_p,
                                do_sample=gen_do_sample,
                            )

                            # Save the first generated sample
                            if generated_texts:
                                svg_filename = (
                                    sample_dir
                                    / f"sample_{i + 1}_step{completed_steps}.txt"
                                )
                                save_svg_text(
                                    generated_texts[0], svg_filename.as_posix()
                                )
                            else:
                                logger.info(
                                    f"[Warning] Generation returned empty list for sample {i + 1}"
                                )

                        logger.info(f"Saved {xcfg.eval_sample} samples to {sample_dir}")
                        # End main process block for eval/save

                    accelerator.wait_for_everyone()
                    model.train()
                    logger.info(
                        f"Step {completed_steps}: Evaluation and saving complete."
                    )
                    # End Evaluation and Checkpointing

            if completed_steps >= max_train_steps:
                break

        # Check if training should end after epoch completes
        if completed_steps >= max_train_steps:
            logger.info(
                f"Reached max_train_steps ({max_train_steps}). Stopping training."
            )
            break

    # Training finished
    logger.info("\nTraining finished. Saving final model...")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_save_dir_base = project_dir / f"final_step_{completed_steps}"

        # Save the standard trained model
        final_model_dir = final_save_dir_base / "model"
        final_model_dir.mkdir(parents=True, exist_ok=True)
        accelerator.print(f"Saving final standard trained model to: {final_model_dir}")

        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            final_model_dir.as_posix(),
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
        )
        tokenizer.save_pretrained(final_model_dir.as_posix())
        accelerator.print(f"Standard model saved.")

        # Save the EMA model weights
        if use_ema and ema_instance:
            final_ema_model_dir = final_save_dir_base / "ema_model"
            final_ema_model_dir.mkdir(parents=True, exist_ok=True)
            accelerator.print(
                f"Saving final EMA model weights to: {final_ema_model_dir}"
            )
            try:
                # Create a temporary model instance or use the unwrapped_model to load EMA weights
                ema_weights = ema_instance.ema_model.state_dict()
                unwrapped_model.load_state_dict(ema_weights)
                unwrapped_model.save_pretrained(
                    final_ema_model_dir.as_posix(),
                    save_function=accelerator.save,
                )
                tokenizer.save_pretrained(final_ema_model_dir.as_posix())
                accelerator.print(f"EMA model saved.")
            except Exception as e:
                accelerator.print(f"[Error] Failed to save EMA model: {e}")

    accelerator.wait_for_everyone()

    # End Training
    progress_bar.close()
    if xcfg.with_tracking:
        accelerator.end_training()
    logger.info("--- Training Complete ---")
