# -*- coding: utf-8 -*-
# Author: ximing
# Description: hydra_utils
# Copyright (c) 2024, XiMing Xing.
# License: MIT License

import hydra
import omegaconf
from omegaconf import open_dict
from omegaconf import OmegaConf


def get_runtime_dir(cfg: omegaconf.DictConfig):
    with open_dict(cfg):  # runtime output directory
        runtime_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    return runtime_dir


def set_cfg_struct(cfg: omegaconf.DictConfig, open: bool):
    if OmegaConf.is_struct(cfg):
        OmegaConf.set_struct(cfg, open)
