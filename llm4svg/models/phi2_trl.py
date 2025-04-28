# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: LLM4SVG(Phi-2) Training and Inference using TRL

import logging
from pathlib import Path
import random
from typing import List, Dict, Any, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from datasets import Dataset as HFDataset
import omegaconf
from tqdm.auto import tqdm
from accelerate import Accelerator
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    PreTrainedTokenizerBase,
    GenerationConfig
)

from llm4svg.utils import model_size
from llm4svg.data import syntactic2svg, SVGTokenizer, TokenDescMapper
from llm4svg.svglib import save_svg_text


def init_token_embedding(
        xcfg: omegaconf.DictConfig,
        model: AutoModelForCausalLM,
        svg_tokenizer: SVGTokenizer,
        logger: logging.Logger,
):
    """Initializes token embeddings, resizing if necessary and applying semantic initialization."""
    logger.info("Initializing token embeddings...")
    new_vocab_size = len(svg_tokenizer)
    embedding_layer = model.get_input_embeddings()
    old_vocab_size = embedding_layer.weight.shape[0]

    if new_vocab_size > old_vocab_size:
        logger.info(f"Resizing token embeddings from {old_vocab_size} to {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)
        # After resizing, update embedding_layer reference if needed, though usually not necessary
        embedding_layer = model.get_input_embeddings()

    # Semantic initialization for *newly added* tokens
    num_new_tokens = new_vocab_size - old_vocab_size
    if num_new_tokens > 0 and xcfg.semantic_init_svg_token:
        new_tokens = svg_tokenizer.get_new_tokens()
        with torch.no_grad():
            # Iterate backwards through the indices of the new tokens in the embedding matrix
            for i, token in enumerate(reversed(new_tokens), start=1):
                # Get description, handle missing keys gracefully
                desc = TokenDescMapper.get(token)  # Use .get for safety
                if desc is None:
                    logger.warning(
                        f"No description found in TokenDescMapper for new token: '{token}'. Skipping semantic init.")
                    continue

                # Tokenize description, ensuring we only use original vocab IDs
                try:
                    # Assuming tokenize_ids exists and returns List[int]
                    tokenized_ids = svg_tokenizer.tokenize_ids(desc)
                    # Filter IDs to only include those from the *original* vocabulary
                    valid_ids = [id_ for id_ in tokenized_ids if id_ < old_vocab_size]
                except Exception as e:
                    logger.error(f"Error tokenizing description for token '{token}': {e}", exc_info=True)
                    continue

                if not valid_ids:
                    logger.warning(
                        f"Description for token '{token}' tokenized to empty list or only new/unknown IDs. Cannot initialize.")
                    continue

                # Calculate mean embedding from valid original tokens
                # Use original embedding layer reference for safety if resize happened
                original_embeddings = model.get_input_embeddings().weight[:old_vocab_size, :]
                new_embedding = original_embeddings[valid_ids, :].mean(axis=0)

                # Assign the calculated embedding to the new token's position
                embedding_layer.weight[-i, :] = new_embedding.clone().detach()

    logger.info("Token embedding initialization finished.")
    return model


class Phi2DataCollator:
    """
    Data collator for Phi-2 SFT that formats prompts and masks labels.
    Assumes input examples have 'text_prompt' and 'svg_text'.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, ignore_index: int = -100):
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        if self.tokenizer.pad_token_id is None:
            # Ensure pad token ID is set for padding
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            print(f"Tokenizer pad_token_id not set. Using eos_token_id: {self.tokenizer.eos_token_id}")

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [f"Instruct: {ex['text_prompt']}\nOutput: " for ex in examples]
        answers = [str(ex['svg_text']) for ex in examples]  # Ensure string conversion

        # Tokenize prompts and answers separately (handle potential errors)
        try:
            # No truncation/padding here
            prompt_tokenized = self.tokenizer(prompts, padding=False, truncation=False)
            answer_tokenized = self.tokenizer(answers, padding=False, truncation=False)
        except Exception as e:
            raise e

        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for i in range(len(examples)):
            prompt_ids = prompt_tokenized['input_ids'][i]
            answer_ids = answer_tokenized['input_ids'][i]

            # Combine prompt and answer
            current_input_ids = prompt_ids + answer_ids
            # Create labels: mask prompt, keep answer
            current_labels = ([self.ignore_index] * len(prompt_ids)) + answer_ids

            # Combine attention masks if they exist
            prompt_mask = prompt_tokenized.get('attention_mask', [1] * len(prompt_ids))[i]
            answer_mask = answer_tokenized.get('attention_mask', [1] * len(answer_ids))[i]
            current_attention_mask = prompt_mask + answer_mask

            # Append to batches (as lists for now)
            input_ids_batch.append(current_input_ids)
            labels_batch.append(current_labels)
            attention_mask_batch.append(current_attention_mask)

        input_ids_padded = pad_sequence(
            [torch.tensor(seq) for seq in input_ids_batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels_padded = pad_sequence(
            [torch.tensor(seq) for seq in labels_batch],
            batch_first=True,
            padding_value=self.ignore_index  # Pad labels with ignore index
        )
        attention_mask_padded = pad_sequence(
            [torch.tensor(seq) for seq in attention_mask_batch],
            batch_first=True,
            padding_value=0  # Pad attention mask with 0
        )

        return {
            'input_ids': input_ids_padded,
            'labels': labels_padded,
            'attention_mask': attention_mask_padded
        }


def llm4svg_phi2_sft_via_trl(
        cfg: omegaconf.DictConfig,
        project_dir: Path,
        accelerator: Accelerator,
        dataset: HFDataset,
        eval_dataset: Optional[HFDataset],
        logger: logging.Logger
):
    """Fine-tunes a Phi-2 model using TRL SFTTrainer."""

    # Redirect to inference if specified in config
    if cfg.x.inference:
        logger.info("Inference mode detected. Redirecting to inference_phi2...")
        inference_phi2(cfg, project_dir, dataset, eval_dataset, logger)
        return

    xcfg, data_cfg = cfg.x, cfg.data

    logger.info("--- Starting LLM4SVG SFT via TRL ---")
    logger.info(f"Project Directory: {project_dir}")
    logger.info(
        f"Key Configs: model_name={xcfg.model_name}, seq_len={xcfg.seq_len}, lr={xcfg.lr}, epochs={xcfg.num_train_epochs}"
    )

    # Load Tokenizer
    logger.info("Loading tokenizer...")
    tokenizer_load_path = xcfg.get("tokenizer_load_path", xcfg.model_name)

    if xcfg.get('use_svg_token', False):
        try:
            tokenizer = SVGTokenizer(xcfg, print_fn=logger.info, local_files_only=xcfg.local_file)
            logger.info(f"Loaded SVGTokenizer from {tokenizer_load_path}")
        except Exception as e:
            logger.warning(f"Could not load SVGTokenizer from {tokenizer_load_path}. Attempting init. Error: {e}")
            tokenizer = SVGTokenizer(xcfg, print_fn=logger.info)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, local_files_only=xcfg.local_file,
                                                  trust_remote_code=True)
        logger.info(f"Loaded AutoTokenizer from {tokenizer_load_path}")

    # Ensure PAD token is set (Phi-2 might not have one)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning(
                f"Tokenizer pad_token not set. Using eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.warning("Tokenizer pad_token and eos_token not set. Added '[PAD]' as pad token.")

    # Load Model
    logger.info(f"Loading model: {xcfg.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        xcfg.model_name,
        local_files_only=xcfg.local_file,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if xcfg.get("use_bf16", False) else torch.float32
    )
    logger.info(f"Model loaded. Config: {model.config}")
    logger.info(f"Model size: {model_size(model, 'M'):.2f}M parameters")

    # Resize model embeddings if tokenizer added tokens
    if len(tokenizer) > model.config.vocab_size:
        logger.info(f"Resizing model embeddings to match tokenizer vocab size: {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # Verify embedding size matches after resize
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
            logger.error("Mismatch between embedding size and tokenizer size after resize!")

    # Initialize SVG Tokens
    if isinstance(tokenizer, SVGTokenizer):
        model = init_token_embedding(xcfg, model, tokenizer, logger)

    # Prepare Datasets
    # The custom collator expects 'text_prompt' and 'svg_text'
    # Ensure these columns exist in your dataset object passed to this function
    required_columns = {'text_prompt', 'svg_text'}
    if not required_columns.issubset(dataset.column_names):
        logger.error(f"Training dataset missing required columns: {required_columns - set(dataset.column_names)}")
        raise ValueError("Training dataset missing required columns.")

    if eval_dataset and not required_columns.issubset(eval_dataset.column_names):
        logger.warning(f"Eval dataset missing required columns: {required_columns - set(eval_dataset.column_names)}. "
                       f"Evaluation might fail.")

    # Log dataset examples (main process only)
    if accelerator.is_main_process:
        num_samples_to_log = 2
        indices = random.sample(range(len(dataset)), min(num_samples_to_log, len(dataset)))
        for index in indices:
            logger.info(f"\n--- Train Sample {index} ---")
            logger.info(f"Text Prompt: {dataset[index]['text_prompt']}")
            logger.info(f"SVG Text: {str(dataset[index]['svg_text'])}")
            logger.info(f"--- End Sample ---\n")

    # Instantiate Data Collator: Pass tokenizer, sequence length handled by SFTTrainer
    data_collator = Phi2DataCollator(tokenizer=tokenizer)

    # Training Arguments
    output_dir = project_dir / "training_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define evaluation strategy based on presence of eval_dataset and config
    eval_strategy = "no"
    eval_steps = None
    if eval_dataset and xcfg.get("eval_steps", 0) > 0:
        eval_strategy = "steps"
        eval_steps = xcfg.eval_steps
        logger.info(f"Evaluation enabled: strategy='{eval_strategy}', steps={eval_steps}")
    else:
        logger.info("Evaluation disabled.")

    # Get resume path
    resume_path = xcfg.get('resume_from_checkpoint', None)
    # Determine the value for TrainingArguments: path string or explicit False
    resume_arg = resume_path if resume_path and Path(resume_path).exists() else False
    if resume_path and not resume_arg:
        logger.warning(f"resume_from_checkpoint path specified but not found: {resume_path}. Starting fresh.")

    training_args = TrainingArguments(
        output_dir=output_dir.as_posix(),

        # resuming
        resume_from_checkpoint=resume_arg,

        # Training Strategy
        num_train_epochs=xcfg.get('num_train_epochs', 3),
        per_device_train_batch_size=xcfg.get('train_batch_size', 1),
        gradient_accumulation_steps=xcfg.get('gradient_accumulation_steps', 2),

        # Optimizer
        optim=xcfg.get('optim', 'adamw_torch'),
        learning_rate=xcfg.get('lr', 5e-5),
        weight_decay=xcfg.get('weight_decay', 0.01),
        adam_beta1=xcfg.get('adam_beta1', 0.9),
        adam_beta2=xcfg.get('adam_beta2', 0.999),
        adam_epsilon=xcfg.get('adam_epsilon', 1e-8),

        # Scheduler
        lr_scheduler_type=xcfg.get('lr_scheduler', 'cosine'),
        warmup_steps=xcfg.get('warmup_steps', 100),

        # Gradient Clipping
        max_grad_norm=xcfg.get('grad_max_norm', 1.0),

        # Logging & Saving
        logging_dir=(project_dir / "logs").as_posix(),
        logging_steps=xcfg.get('logging_steps', 10),
        save_strategy="steps",
        save_steps=xcfg.get('save_steps', 500),
        save_total_limit=xcfg.get('save_total_limit', 5),

        # Evaluation
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        per_device_eval_batch_size=xcfg.get('eval_batch_size', xcfg.get('train_batch_size', 1)),

        # Mixed Precision
        fp16=xcfg.get("use_fp16", False),
        bf16=xcfg.get("use_bf16", False),

        # System
        dataloader_num_workers=data_cfg.get('num_workers', 0),
        report_to=xcfg.get('report_to', "none"),
        remove_unused_columns=False,
        seed=cfg.seed,
    )

    sft_trainer_kwargs = dict(
        max_seq_length=xcfg.get('seq_len', 512),
        packing=True,
        dataset_text_field="text_prompt",  # Or "svg_text" or another valid text column
        dataset_kwargs={'skip_prepare_dataset': True},
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset if eval_strategy != "no" else None,
        data_collator=data_collator,
        # Pass SFT args directly using dictionary expansion
        **sft_trainer_kwargs,
    )

    # Train
    logger.info("Starting training...")
    try:
        # train_result = trainer.train(resume_from_checkpoint=xcfg.get('resume_from_checkpoint', None))
        train_result = trainer.train()
        logger.info("Training finished.")

        # Save Final Model, State, and Tokenizer
        final_checkpoint_dir = output_dir / "final_checkpoint"
        logger.info(f"Saving final model and tokenizer to {final_checkpoint_dir}...")
        trainer.save_model(final_checkpoint_dir.as_posix())
        try:
            tokenizer.save_pretrained(final_checkpoint_dir.as_posix())
        except Exception as e:
            logger.error(f"Could not save tokenizer: {e}", exc_info=True)

        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()  # Save trainer state (optimizer, scheduler, etc.)
        logger.info("Final model, tokenizer, metrics, and state saved.")

    except Exception as e:
        logger.error("Error during training!", exc_info=True)
        raise e

    accelerator.wait_for_everyone()  # Ensure all processes finish before exiting
    logger.info("--- Phi-2 SFT Training Script Finished ---")


def inference_phi2(
        cfg: omegaconf.DictConfig,
        project_dir: Path,  # Keep project_dir for saving samples
        dataset: Optional[HFDataset],  # dataset might not be needed if using eval_dataset
        eval_dataset: Optional[HFDataset],  # Make optional
        logger: logging.Logger  # <<< Accept standard logger
):
    """Performs inference using a trained Phi-2 model."""
    logger.info("--- Starting Phi-2 Inference ---")
    xcfg = cfg.x
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference device: {device}")

    checkpoint_path = xcfg.get('resume_from_checkpoint')
    if not checkpoint_path:
        logger.error("Missing 'resume_from_checkpoint' in config for inference.")
        return
    logger.info(f"Loading model and tokenizer from checkpoint: {checkpoint_path}")

    # --- Load Tokenizer (from checkpoint) ---
    try:
        # Check if it was SVGTokenizer or AutoTokenizer during saving
        # Safest is usually AutoTokenizer.from_pretrained on the checkpoint dir
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        # Re-apply padding token logic if needed (sometimes not saved)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.warning(f"Tokenizer pad_token not set after loading. Using eos_token: {tokenizer.eos_token}")
            else:
                # This case should be rare if saved correctly
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logger.warning("Tokenizer pad_token and eos_token not set after loading. Added '[PAD]'.")
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {checkpoint_path}: {e}", exc_info=True)
        return

    # Mixed Precision Flags
    use_bf16 = xcfg.get("use_bf16", False)
    use_fp16 = xcfg.get("use_fp16", False)
    if use_bf16 and use_fp16:
        logger.warning("Both use_bf16 and use_fp16 are True. Prioritizing BF16 for A800.")
        use_fp16 = False
    model_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    logger.info(f"Using model dtype: {model_dtype}")

    # Load Model (from checkpoint)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,  # Add trust_remote_code=True
            torch_dtype=model_dtype,
            device_map="auto",
        )
        # Resize embeddings if tokenizer has extra tokens (e.g., added pad token)
        if len(tokenizer) > model.config.vocab_size:
            logger.info(f"Resizing model embeddings during inference to match tokenizer: {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        model.to(device)
        model.eval()  # Set model to evaluation mode
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model from {checkpoint_path}: {e}", exc_info=True)
        return

    # Get Generation Config
    gen_cfg = xcfg.get('generation', {})
    max_new_tokens = gen_cfg.get('max_new_tokens', xcfg.get('seq_len', 1024))  # Use seq_len as fallback default
    do_sample = gen_cfg.get('do_sample', True)
    temperature = gen_cfg.get('temperature', 0.7)
    top_p = gen_cfg.get('top_p', 0.9)
    top_k = gen_cfg.get('top_k', 50)

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    logger.info(f"Generation config: {generation_config}")

    # Prepare Dataset for Inference
    inference_dataset = eval_dataset if eval_dataset else dataset  # Prefer eval_dataset
    if not inference_dataset:
        logger.error("No dataset provided for inference.")
        return
    if 'text_prompt' not in inference_dataset.column_names:
        logger.error("Inference dataset missing 'text_prompt' column.")
        return

    # Create Output Directory
    sample_dir = project_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving generated samples to: {sample_dir}")

    # Inference Loop
    # No need for DataCollator here
    num_examples = len(inference_dataset)
    logger.info(f"Generating samples for {num_examples} examples...")
    for i, example in enumerate(tqdm(inference_dataset, desc="Generating Samples")):
        # Use logger.info for detailed logging, tqdm for progress
        # logger.info(f"Generating sample {i + 1}/{num_examples}")
        text_prompt = f"Instruct: {example['text_prompt']}\nOutput: "

        try:
            # Tokenize the single prompt
            inputs = tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=xcfg.get('seq_len',
                                                                                                      512) - max_new_tokens)  # Truncate prompt leaving space for new tokens
            inputs = inputs.to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, generation_config=generation_config)

            # Decode the full output (including prompt)
            # Decode only the generated part: outputs[0, inputs['input_ids'].shape[1]:]
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]  # Get only new tokens
            generated_svg_text = tokenizer.decode(generated_ids,
                                                  skip_special_tokens=True)  # Skip special tokens for cleaner output

            logger.info(f"\n--- Sample {i + 1} ---")
            logger.info(f"Prompt: {example['text_prompt']}")
            logger.info(f"Generated SVG Text: {generated_svg_text}")

            # Save the generated text
            txt_filename = sample_dir / f"svg_{i + 1}.txt"
            save_svg_text(generated_svg_text, txt_filename.as_posix())

            # Convert to SVG and save (handle potential errors)
            try:
                svg_code = syntactic2svg(generated_svg_text)
                svg_filename = sample_dir / f"svg_{i + 1}.svg"
                save_svg_text(svg_code, svg_filename.as_posix())
                # logger.debug(f"Generated SVG code (first 100 chars): {svg_code[:100]}...") # Optional debug
            except Exception as conversion_err:
                logger.error(f"Error converting generated text to SVG for sample {i + 1}: {conversion_err}")

        except Exception as gen_err:
            logger.error(f"Error during generation for sample {i + 1}: {gen_err}", exc_info=True)
            # Continue to next sample or break? Continue for now.

    logger.info(f"--- Phi-2 Inference Finished. Samples saved in {sample_dir} ---")
