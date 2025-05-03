# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License

import logging
from pathlib import Path

import hydra
import omegaconf
from accelerate.utils import set_seed

from llm4svg.data import get_svg_dataset
from llm4svg.models import *
from llm4svg.utils import (
    LogUtilsMixinAccelerator,
    get_runtime_dir,
    initialize_accelerator,
    set_cfg_struct,
)


@hydra.main(version_base=None, config_path="configs", config_name="hydra")
def main(cfg: omegaconf.DictConfig):
    set_cfg_struct(cfg, False)  # Allow dynamic parameter addition

    # Seed
    set_seed(
        cfg.seed, device_specific=cfg.device_specific, deterministic=cfg.deterministic
    )

    # Project dir
    project_dir = Path(get_runtime_dir(cfg))

    # Init accelerator
    accelerator = initialize_accelerator(cfg.x, project_dir.as_posix())

    # Initialize Logger
    logger_util = LogUtilsMixinAccelerator(
        accelerator,
        project_dir,
        level=logging.getLevelName(cfg.logging.get("level", "INFO").upper()),
        log_file_basename=cfg.logging.get("log_file_basename", "training_run"),
        log_all_ranks_console=cfg.logging.get("log_all_ranks_console", False),
    )
    actual_logger = logger_util.get_logger()

    # Download SVG dataset
    train_dataset, eval_dataset = get_svg_dataset(
        cfg.data, cfg.x.data_preprocess, logger=actual_logger
    )
    accelerator.wait_for_everyone()

    # Methods mapping
    method_map = {
        "LLM4SVG-GPT2": lambda: llm4svg_gpt2_sft(
            cfg, project_dir, accelerator, train_dataset, actual_logger
        ),
        "LLM4SVG-Phi2": lambda: llm4svg_phi2_sft_via_trl(
            cfg, project_dir, accelerator, train_dataset, eval_dataset, actual_logger
        ),
        # "LLM4SVG-Llama3-unsloth": lambda: llama3_sft_by_unsloth(cfg, project_dir, accelerator, train_dataset,
        #                                                         eval_dataset, actual_logger),
    }

    method = cfg.get("x", {}).get("method", "")
    if method not in method_map:
        raise NotImplementedError(f"Method {method} not implemented")

    try:
        method_map[method]()
    except Exception as e:
        logging.error(f"Failed to execute method {method}: {e}")
        return


if __name__ == "__main__":
    main()
