# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License

from .svg_dataset import get_svg_dataset
from .semantic_tokens import PathCMDMapper, AttribMapper, ContainerMapper, PathMapper, GradientsMapper, ShapeMapper, \
    SVGToken, syntactic2svg
from .tokenizer import SVGTokenizer, NUM_TOKEN
from .token_config import SEMANTIC_SVG_TOKEN_MAPPER_DEFAULT as TokenDescMapper, SEMANTIC_SVG_TOKEN_MAPPER_ADVANCE
