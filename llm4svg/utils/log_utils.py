# -*- coding: utf-8 -*-
# Author: ximing
# Description: log_utils
# Copyright (c) 2024, XiMing Xing.
# License: MIT License

import logging
from pathlib import Path
import sys

import datasets
import transformers
from accelerate import Accelerator


class LogUtilsMixinAccelerator:

    def __init__(
            self,
            accelerator: Accelerator,
            project_dir: Path,
            level=logging.INFO,
            log_file_basename='main_process',  # Log file name for main rank
            # formatter_str='%(asctime)s - RANK %(process_index)s - [%(levelname)s] - %(name)s - %(message)s',
            formatter_str='%(asctime)s - RANK %(process_index)s - [%(levelname)s] - %(message)s',
            log_all_ranks_console: bool = False
    ):
        self.accelerator = accelerator
        self.log_all_ranks_console = log_all_ranks_console

        # Use a unique logger name per instance
        self._logger = logging.getLogger(f"{__name__}.rank_{accelerator.process_index}")
        self._logger.setLevel(level)
        self._logger.propagate = False

        # Formatter
        formatter = logging.Formatter(formatter_str, defaults={"process_index": accelerator.process_index})

        # Console Handlers (Added to ALL processes initially)
        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)  # Base level for stdout
        stdout_handler.addFilter(lambda record: record.levelno <= logging.INFO)
        stdout_handler.setFormatter(formatter)

        stderr_handler = logging.StreamHandler(stream=sys.stderr)
        stderr_handler.setLevel(logging.WARNING)  # Base level for stderr
        stderr_handler.setFormatter(formatter)

        self._logger.addHandler(stdout_handler)
        self._logger.addHandler(stderr_handler)

        console_log_active_msg = "ENABLED"
        # Conditionally Silence Console Handlers on Non-Main Ranks
        if not accelerator.is_main_process and not self.log_all_ranks_console:
            # If NOT main process AND the flag to log all ranks is FALSE,
            # silence the console handlers for this rank.
            silence_level = logging.CRITICAL + 1  # Level higher than any standard log
            stdout_handler.setLevel(silence_level)
            stderr_handler.setLevel(silence_level)
            console_log_active_msg = "DISABLED (Use log_all_ranks_console=True to enable)"

        # File Handler (MAIN process ONLY)
        file_log_active_msg = "DISABLED"
        file_handler = None  # Initialize file_handler to None
        if accelerator.is_main_process:
            log_file_name = f'{log_file_basename}.log'
            log_file_path = project_dir / log_file_name
            project_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file_path.as_posix(), mode='a')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)  # File handler logs at the specified level
            self._logger.addHandler(file_handler)
            file_log_active_msg = f"ENABLED: {log_file_path} (Level: {logging.getLevelName(file_handler.level)})"

        # Configure Library Verbosity (Main process more verbose)
        main_process_level = level
        non_main_process_level = logging.WARNING  # Or logging.ERROR

        # Use stored levels, respecting the console silencing logic for libraries too?
        # Setting library verbosity high here might override console silencing for library logs.
        # It's usually better to control application log verbosity via the main flag.
        # Let's keep library verbosity linked to main process status primarily.
        effective_transformer_level = main_process_level if accelerator.is_main_process else non_main_process_level
        effective_datasets_level = main_process_level if accelerator.is_main_process else non_main_process_level

        self.set_transformers_verbosity_level(effective_transformer_level)
        self.set_datasets_verbosity_level(effective_datasets_level)

        # Log Initial Configuration
        self._logger.info(f"--- Logger Initialized for RANK {accelerator.process_index} ---")
        self._logger.info(f"  - Overall logger level set to: {logging.getLevelName(self._logger.level)}")
        self._logger.info(f"  - Console Logging: {console_log_active_msg}")
        self._logger.info(f"     - stdout Level (effective): {logging.getLevelName(stdout_handler.level)}")
        self._logger.info(f"     - stderr Level (effective): {logging.getLevelName(stderr_handler.level)}")
        self._logger.info(f"  - File Logging: {file_log_active_msg}")
        self._logger.info(
            f"  - Transformers verbosity set to: {logging.getLevelName(transformers.utils.logging.get_verbosity())}")
        self._logger.info(
            f"  - Datasets verbosity set to: {logging.getLevelName(datasets.utils.logging.get_verbosity())}")
        self._logger.info(f"--- End Logger Initialized ---")

    def get_logger(self):
        """Returns the configured logger instance for this process."""
        return self._logger

    # Adjusted verbosity setting methods to accept a level parameter
    def set_transformers_verbosity_level(self, level):
        """Sets the verbosity level for the transformers library logger."""
        transformers.utils.logging.set_verbosity(level)

    def set_datasets_verbosity_level(self, level):
        """Sets the verbosity level for the datasets library logger."""
        datasets.utils.logging.set_verbosity(level)

    # Convenience methods (info/debug log directly using the logger)
    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    # --- Main Process Only Logging Methods ---
    def info_main(self, msg, *args, **kwargs):
        if self.accelerator.is_main_process:
            self._logger.info(msg, *args, **kwargs)

    def debug_main(self, msg, *args, **kwargs):
        if self.accelerator.is_main_process:
            self._logger.debug(msg, *args, **kwargs)

    def warning_main(self, msg, *args, **kwargs):
        if self.accelerator.is_main_process:
            self._logger.warning(msg, *args, **kwargs)

    def error_main(self, msg, *args, **kwargs):
        if self.accelerator.is_main_process:
            self._logger.error(msg, *args, **kwargs)
