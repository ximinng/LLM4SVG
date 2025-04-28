# -*- coding: utf-8 -*-
# Author: ximing
# Description: train_utils
# Copyright (c) 2024, XiMing Xing.
# License: MIT License

import math
import os
from typing import Union

import omegaconf
from accelerate import Accelerator
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def initialize_accelerator(cfg: omegaconf.DictConfig, project_dir: Union[str, os.PathLike]) -> Accelerator:
    # Initialize the accelerator with optional mixed precision and gradient accumulation settings
    report_to = cfg.get('report_to', None)
    accelerator = Accelerator(
        mixed_precision=cfg.get('mixed_precision', 'no'),
        gradient_accumulation_steps=cfg.get('gradient_accumulation_steps', 1),
        project_dir=project_dir,
        log_with=report_to if cfg.get('with_tracking', False) else None
    )
    accelerator.wait_for_everyone()  # Synchronize all processes before proceeding

    return accelerator


def model_size(model: nn.Module, unit=""):
    if unit == 'M':  # Million
        return sum(t.numel() for t in model.parameters()) / 1000 ** 2
    elif unit == 'B':  # Billion
        return sum(t.numel() for t in model.parameters()) / 1000 ** 3
    else:
        return sum(t.numel() for t in model.parameters())


def create_adam_optimizer(module_params, lr: float, weight_decay: float, beta_1: float = 0.9, beta_2: float = 0.999,
                          epsilon: float = 1e-8, split_decay_params: bool = False):
    """Creates an AdamW optimizer with optional parameter grouping for weight decay."""

    if not isinstance(module_params, nn.Module):
        # If input is already parameters, we can't easily do named_parameters splitting
        # Fall back to standard AdamW or raise an error if splitting was requested
        if split_decay_params:
            print(
                "[Warning] split_decay_params=True requires an nn.Module instance. Applying weight decay to all parameters."
            )
            # raise ValueError("split_decay_params=True requires an nn.Module instance.")
        optimizer = torch.optim.AdamW(module_params, lr=lr, weight_decay=weight_decay, eps=epsilon,
                                      betas=(beta_1, beta_2))
        return optimizer

    if split_decay_params:
        no_decay = ["bias", "layer_norm.weight", "layernorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in module_params.named_parameters() if
                           p.requires_grad and not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in module_params.named_parameters() if
                           p.requires_grad and any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,  # Explicitly set weight decay to 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr, betas=(beta_1, beta_2), eps=epsilon)
    else:
        # Apply weight decay to all parameters
        optimizer = torch.optim.AdamW(module_params.parameters(), lr=lr, weight_decay=weight_decay, eps=epsilon,
                                      betas=(beta_1, beta_2))

    return optimizer


def compute_train_schedule(
        train_dataloader: DataLoader,
        num_train_epochs: int,
        gradient_accumulation_steps: int = 1,
        resume_step: int = 0
):
    # Calculate the number of update steps per epoch
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

    # Calculate the maximum training steps based on epochs and update steps
    max_train_steps = num_train_epochs * num_update_steps_per_epoch

    # Calculate completed steps and starting epoch based on resume_step
    completed_steps = resume_step
    starting_epoch = resume_step // num_update_steps_per_epoch

    # Adjust max_train_steps if resuming from a specific step
    if resume_step > 0:
        max_train_steps = max(0, max_train_steps - resume_step)

    # Recalculate number of epochs needed based on the adjusted max_train_steps
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch) if max_train_steps > 0 else 0

    return max_train_steps, num_train_epochs, completed_steps, starting_epoch
