# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: SVGX-Dataset

import logging
from pathlib import Path
from typing import Tuple, Union

import omegaconf
from datasets import Dataset, DatasetDict, disable_caching, load_dataset, load_from_disk

from llm4svg.data.semantic_tokens import svg2syntactic
from llm4svg.svglib import (
    apply_g_attributes_to_children,
    remove_svg_tag,
    replace_gradient_tags,
)
from llm4svg.utils import path_exists

disable_caching()


def get_svg_dataset(
    cfg: omegaconf.DictConfig, preprocess: bool, logger: logging.Logger
) -> Tuple[Union[Dataset, DatasetDict], Union[Dataset, DatasetDict]]:
    if path_exists(cfg.preprocessed_disk):
        processed_dataset = load_from_disk(cfg.preprocessed_disk)
    else:
        disable_caching()

        assert set(cfg.text_prompt).issubset(
            set(cfg.target_columns)
        ), "target_columns must contains all text prompts"

        dataset_name = cfg.dataset

        local_dir = cfg.get("load_from_disk", None)
        if local_dir and Path(local_dir).exists():
            # load dataset
            dataset = load_from_disk(local_dir)
        else:
            # download dataset
            dataset = load_dataset(
                dataset_name,
                split=cfg.split,
                token=cfg.HG_auth_token,
                cache_dir=cfg.cache_dir,
            )
            dataset.save_to_disk(cfg.cache_dir)

        logger.info(
            f"Load dataset {dataset_name}, column names: {dataset.column_names}"
        )
        logger.info(f"Select split: {cfg.split}, found {len(dataset)} samples")

        # remove unused columns
        if cfg.remove_columns:
            dataset = dataset.remove_columns(cfg.unused_columns)
        logger.info(f"Select dataset column_names: {dataset.column_names}")

        def preprocess_data(example):
            struct_svg = "no struct_svg"
            svg_str = example["svg_code"]

            try:
                # Remove the gradient tag to simplify the color representation
                if cfg.simplify_grad_tag:
                    svg_str = replace_gradient_tags(svg_str, cfg.fill_is_empty)

                # Delete the <svg> tag and keep the other tags
                if cfg.remove_svg:
                    svg_str = remove_svg_tag(svg_str)

                # Flatten and inherit group
                if cfg.flatten_g:
                    svg_str = apply_g_attributes_to_children(svg_str)

                # SVG string to syntactic representation
                if cfg.syntactic_encode:
                    struct_svg, svg_desc = svg2syntactic(svg_str)
                    svg_str = svg_desc
            except Exception as e:
                logger.info(e)
                logger.info(example["source"])
                logger.info(example["svg_path"])

            text_prompt = ". ".join([example[i] for i in cfg.text_prompt])
            image = {"image": example["image"]} if "image" in cfg.target_columns else {}

            # 1. regular return
            output = {"text_prompt": text_prompt, "svg_text": svg_str, **image}
            if not cfg.debug_data:
                return output
            # 2. debug return
            output["svg_raw"] = example["svg_code"]  # unprocessed svg
            output["struct_svg"] = struct_svg  # printable SVG with formatting structure
            return output

        if preprocess:
            processed_dataset = dataset.map(
                preprocess_data,
                num_proc=cfg.get("num_workers", 1),
                remove_columns=cfg.target_columns,
                load_from_cache_file=cfg.get("load_from_cache_file", False),
                desc="Train Dataset Generation",
            )
            logger.info(f"num of processed dataset: {len(processed_dataset)}")
        else:
            processed_dataset = dataset

        if cfg.preprocessed_disk is not None:
            processed_dataset.save_to_disk(cfg.preprocessed_disk)

    # Clean up the Arrow cache files in the directory
    processed_dataset.cleanup_cache_files()

    # Create validation dataset
    if cfg.get("for_recon", False):
        # To evaluate reconstruction performance, is a subset of the training set
        eval_size = min(max(0, int(cfg.eval_size)), len(processed_dataset))
        train_dataset, eval_dataset = processed_dataset, Dataset.from_dict(
            processed_dataset[:eval_size]
        )
    else:
        dataset = processed_dataset.train_test_split(test_size=cfg.eval_size)
        train_dataset, eval_dataset = dataset["train"], dataset["test"]

    return train_dataset, eval_dataset
