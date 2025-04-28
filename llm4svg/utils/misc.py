# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: Helper Functions

from typing import Callable
from pathlib import Path

from torch import is_tensor
from torch.utils._pytree import tree_map


# helper functions

def path_exists(p):
    if p is not None and p != '' and Path(p).exists():
        return True
    return False


def exists(v):
    return v is not None


def default(v, d):
    return v if exists(v) else d


def identity(t):
    return t


def first(it):
    return it[0]


def prepend(arr, el):
    arr.insert(0, el)


def join(arr, delimiter=''):
    return delimiter.join(arr)


def divisible_by(num, den):
    return (num % den) == 0


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def tree_map_tensor(sample, fn: Callable):
    """
    This code defines a tree_map_tensor function that recursively traverses a given sample data structure
    and applies a function fn to all tensors in it.

    Args:
        sample: a sample data structure
        fn: a function

    Examples:
        import torch

        def move_to_gpu(tensor):
            return tensor.to('cuda')

        sample = {
            "input": torch.tensor([1, 2, 3]),
            "metadata": {
                "info": "some_info",
                "image": torch.tensor([4, 5, 6])
            }
        }

        result = tree_map_tensor(sample, move_to_gpu)

    """
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), sample)


def write_lines_to_file(file_path, sentences):
    """
    Write a list of sentences to a text file, one sentence per line.

    Args:
        file_path (str): Path to the text file to write.
        sentences (list): List of sentences (strings) to write.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')


def read_lines_from_file(file_path):
    """
    Read sentences from a text file where each line is a sentence.

    Args:
        file_path (str): Path to the text file to read.

    Returns:
        list: A list of sentences (strings), one sentence per line.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    # Strip any trailing newline characters
    sentences = [sentence.strip() for sentence in sentences]

    return sentences
