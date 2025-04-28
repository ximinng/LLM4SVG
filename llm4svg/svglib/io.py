# -*- coding: utf-8 -*-
# Author: ximing
# Description: io
# Copyright (c) 2024, XiMing Xing.
# License: MIT License

from typing import Union, List

from lxml import etree


def save_svg_string(svg_content: str, filename: str):
    root = etree.fromstring(svg_content.encode('utf-8'))
    tree = etree.ElementTree(root)
    tree.write(filename, pretty_print=True, xml_declaration=True, encoding='UTF-8')


def save_svg_text(svg_content: Union[str, List[str]], filename: str):
    with open(filename, "w", encoding="utf-8") as file:

        if isinstance(svg_content, list):
            for text in svg_content:
                file.writelines(text)
        else:
            file.write(svg_content)
