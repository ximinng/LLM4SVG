# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: SVG Semantic Tokens encoder and decoder

import re
from typing import Union, List, Tuple

from lxml import etree

"""Define SVG Mappers and Identifiers"""

PathCMDMapper = {
    '[m]': '[<|moveto|>]',
    '[l]': '[<|lineto|>]',
    '[h]': '[<|horizontal_lineto|>]',
    '[v]': '[<|vertical_lineto|>]',
    '[c]': '[<|curveto|>]',
    '[s]': '[<|smooth_curveto|>]',
    '[q]': '[<|quadratic_bezier_curve|>]',
    '[t]': '[<|smooth_quadratic_bezier_curveto|>]',
    '[a]': '[<|elliptical_Arc|>]',
    '[z]': '[<|close_the_path|>]',
}

PathCMDIdentifier = {
    'moveto': 'm',
    'lineto': 'l',
    'horizontal_lineto': 'h',
    'vertical_lineto': 'v',
    'curveto': 'c',
    'smooth_curveto': 's',
    'quadratic_bezier_curve': 'q',
    'smooth_quadratic_bezier_curveto': 't',
    'elliptical_Arc': 'a',
    'close_the_path': 'z',
}

AttribMapper = {
    '_id_': '[<|tag_id|>]',
    '_d_': '[<|path_d|>]',
    '_fill_': '[<|fill|>]',
    '_stroke-width_': '[<|stroke-width|>]',
    '_stroke-linecap_': '[<|stroke-linecap|>]',
    '_stroke_': '[<|stroke|>]',
    '_opacity_': '[<|opacity|>]',
    '_transform_': '[<|transform|>]',
    '_gradientTransform_': '[<|gradientTransform|>]',
    '_offset_': '[<|offset|>]',
    '_width_': '[<|width|>]',
    '_height_': '[<|height|>]',
    '_cx_': '[<|cx|>]',
    '_cy_': '[<|cy|>]',
    '_rx_': '[<|rx|>]',
    '_ry_': '[<|ry|>]',
    '_r_': '[<|r|>]',
    '_points_': '[<|points|>]',
    '_x1_': '[<|x1|>]',
    '_y1_': '[<|y1|>]',
    '_x2_': '[<|x2|>]',
    '_y2_': '[<|y2|>]',
    '_x_': '[<|x|>]',
    '_y_': '[<|y|>]',
    '_fr_': '[<|fr|>]',
    '_fx_': '[<|fx|>]',
    '_fy_': '[<|fy|>]',
    '_href_': '[<|href|>]',
    '_rotate_': '[<|rotate|>]',
    '_font-size_': '[<|font-size|>]',
    '_font-style_': '[<|font-style|>]',
    '_font-family_': '[<|font-family|>]',
    '_text-content_': '[<|text-content|>]'
}

SVGToken = {
    'start': '[<|START_OF_SVG|>]',
    'end': '[<|END_OF_SVG|>]',
}

ContainerMapper = {
    '<svg>': SVGToken['start'],
    '</svg>': SVGToken['end'],
    '<g>': '[<|start_of_g|>]',
    '</g>': '[<|end_of_g|>]',
}

ContainerTagIdentifiers = {
    'svg_start': 'START_OF_SVG',
    'svg_end': 'END_OF_SVG',
    'g_start': 'start_of_g',
    'g_end': 'end_of_g'
}

PathMapper = {'<path>': '[<|svg_path|>]'}

PathIdentifier = 'svg_path'

GradientsMapper = {
    '<linearGradient>': '[<|svg_linearGradient|>]',
    '<radialGradient>': '[<|svg_radialGradient|>]',
    '<stop>': '[<|svg_stop|>]',
}

GradientIdentifier = {
    'linear_gradient': 'svg_linearGradient',
    'radial_gradient': 'svg_radialGradient',
    'stop': 'svg_stop'
}

ShapeMapper = {
    '<circle>': '[<|svg_circle|>]',
    '<rect>': '[<|svg_rect|>]',
    '<ellipse>': '[<|svg_ellipse|>]',
    '<polygon>': '[<|svg_polygon|>]',
    '<line>': '[<|svg_line|>]',
    '<polyline>': '[<|svg_polyline|>]',
    '<text>': '[<|svg_text|>]',
}

ShapeIdentifier = {
    'svg_circle': 'circle',
    'svg_rect': 'rect',
    'svg_ellipse': 'ellipse',
    'svg_polygon': 'polygon',
    'svg_line': 'line',
    'svg_polyline': 'polyline',
    'svg_text': 'text',
}


def remove_square_brackets(s: str) -> str:
    """Remove square brackets from the input string."""
    return s.replace('[', '').replace(']', '')


def is_path_closed(path_d: str) -> bool:
    path_d = path_d.strip()
    return path_d.endswith('Z') or path_d.endswith('z')


def svg2syntactic(
        svg_string: str,
        include_gradient_tag: bool = False,
        path_only: bool = False,
        include_group: bool = True,
        group_attr_inherit: bool = True,
        ignore_tags: List[str] = ['clipPath'],
        ignore_attrs: List[str] = ['clip-path', 'gradientUnits'],
) -> Tuple[str, str]:
    """
    Converts an SVG string into a structured text-based representation and a simplified description.

    This function parses SVG XML elements and recursively builds a syntactic tree-like structure,
    extracting shape, path, and gradient information. It supports path-only filtering,
    attribute inheritance from <g> tags, and optional gradient extraction.

    Args:
        svg_string (str): The raw SVG content as a string.
        include_gradient_tag (bool): Whether to include gradient details in the output.
        path_only (bool): If True, only <path> elements will be processed.
        include_group (bool): If True, the <g> (group) tags will be included in the structure output.
        group_attr_inherit (bool): If True, attributes of <g> elements will be inherited by their children.
        ignore_tags (List[str]): List of SVG tag names to ignore during parsing.
        ignore_attrs (List[str]): List of attribute names to ignore during parsing.

    Returns:
        Tuple[str, str]: A tuple with:
            - `struct_ret`: A syntactic structured text representation of the SVG.
            - `svg_desc_ret`: A semantic description string derived from the structured output.
    """
    tree = etree.fromstring(svg_string)

    struct_ret = ""  # Output structure string
    shape_tags = ['circle', 'rect', 'ellipse', 'polygon', 'line', 'polyline']
    gradient_tags = ['linearGradient', 'radialGradient', 'stop']
    stop_attrs = ['offset', 'stop-color']
    gradients = {}  # Stores gradient information
    basic_shape_attrs = [
        'fill', 'stroke-width', 'stroke', 'opacity', 'transform',
        'cx', 'cy', 'r', 'rx', 'ry',
        'width', 'height', 'points',
        'x1', 'y1', 'x2', 'y2',
        'x', 'y', 'dx', 'dy', 'rotate', 'font-size',
        'textLength', 'font-style', 'font-family'
    ]

    def recursive_parse(element, level=0, inherited_attributes=None):
        """Recursively traverses the SVG XML structure and builds a formatted description."""
        nonlocal struct_ret

        tag = etree.QName(element).localname
        inherited_attributes = inherited_attributes or {}

        if tag in ignore_tags:
            return

        # Handle <g> tag: attribute inheritance and optional inclusion
        current_attributes = inherited_attributes.copy()
        if tag == 'g' and group_attr_inherit:
            # Merge 'g' attributes with inherited attributes
            current_attributes.update(element.attrib)
            # Only add the 'g' tag if include_g is True
            if include_group:
                struct_ret += "  " * level + f"<{tag}> (_{current_attributes}_)\n"

            for child in element:
                recursive_parse(child, level + 1, current_attributes)

            if include_group:
                struct_ret += "  " * level + f"</{tag}>\n"
            return  # Skip the rest for `g` tag processing

        # Add current tag to structure (unless gradient or ignored tag)
        if tag not in (*gradient_tags, *ignore_tags):
            struct_ret += "  " * level + f"<{tag}>\n"

        # Get the current element's attributes (merged with inherited)
        attributes = {**element.attrib, **current_attributes}

        # Process <path> tag
        if tag == "path" and "d" in attributes:
            struct_ret += "  " * (level + 1) + f"_d_"

            path_d = attributes['d']
            # parse d attr
            struct_ret += _parse_path_d(path_d, level + 2, float_coords=True)

            is_closed = is_path_closed(path_d)
            # path and its gradient element if any:
            # 1. Non-closed form paths use stroke attr to indicate colors
            # 2. And closed-form paths use fill attr to indicate color
            gradient_id = attributes.get('fill')
            if gradient_id and (gradient_id in gradients) and include_gradient_tag:
                struct_ret += "\n" + "  " * (level + 1) + f"{gradients[gradient_id]}\n"
            elif is_closed:
                struct_ret += f"{_gather_path_attr(attributes)}\n"
            elif not is_closed:
                struct_ret += f"{_gather_path_attr(attributes)}\n"

        # Process <text> tag
        elif tag == "text":
            text_str = element.text if element.text else ""  # get text content

            attributes = element.attrib  # get text attributes
            attr_str = ""
            for attr, value in attributes.items():
                if attr in ['x', 'y', 'dx', 'dy', 'rotate', 'font-size', 'textLength', 'font-style']:
                    attr_str += f"_{attr}_{value}"

            struct_ret += "  " * (level + 1) + f"{attr_str}_text-content_{text_str}\n"

        # Process shape tags (if not in path_only mode)
        # tag: <circle>, <rect>, <ellipse>, <polygon>, <line>, <polyline>
        elif tag in shape_tags and (not path_only):
            point_attrs = {'point'}
            number_attrs = {'cx', 'cy', 'rx', 'ry', 'r', 'x1', 'y1', 'x2', 'y2', 'x', 'y', 'dx', 'dy',
                            'rotate', 'font-size', 'textLength', 'stroke-width', 'opacity'}

            attr_str = ""
            for attr, value in attributes.items():
                if attr in basic_shape_attrs and attr and attr not in ignore_attrs:
                    if attr in point_attrs:
                        attr_str += f"_{attr}_{_parse_point_attr(value)} "
                    elif attr in number_attrs:
                        attr_str += f"_{attr}_{_parse_number(value)} "
                    else:
                        attr_str += f"_{attr}_{value} "

            struct_ret += "  " * (level + 1) + attr_str + "\n"

        # Process gradient-related tags
        # tag: <linearGradient> <radialGradient> <stop>
        elif tag in gradient_tags:
            gradient_id = attributes.get('id')
            if gradient_id:
                gradient_info = f"<{tag}> " + ' '.join(
                    f"_{attr}_{'#' + value if attr == 'id' else value}"
                    for attr, value in attributes.items()
                    if attr not in ['gradientUnits', *ignore_attrs]
                )

                # Handle xlink:href references
                xlink_href = attributes.get("{http://www.w3.org/1999/xlink}href")
                if xlink_href:
                    referenced_id = xlink_href.split("#")[-1]
                    if f"url(#{referenced_id})" in gradients:
                        gradients[f"url(#{gradient_id})"] = gradient_info + gradients[f'url(#{referenced_id})']
                else:
                    gradients[f"url(#{gradient_id})"] = gradient_info

                # Process <stop> inside gradients
                for child in element:
                    if etree.QName(child).localname == 'stop':
                        stop_info = "<stop>" + ' '.join(
                            f"_{attr}_{value}" for attr, value in child.attrib.items()
                            if attr in stop_attrs
                        ) + " </stop>"
                        gradients[f"url(#{gradient_id})"] += stop_info

        # Recursively process child elements
        for child in element:
            recursive_parse(child, level + 1, current_attributes)

        # Add closing tag for root-level <svg> (if applicable)
        if tag in ['svg']:
            struct_ret += "  " * level + f"</{tag}>\n"

    # Begin parsing from root <svg> tag
    recursive_parse(tree)

    # Post-processing: flatten and describe
    flatten_struct_ret = struct_ret.replace("\n", "")  # remove '\n'
    svg_desc_ret = _to_svg_description(flatten_struct_ret)  # to svg description

    return struct_ret, _clean_svg_desc_output(svg_desc_ret)


def _parse_number(num_str: str, round_num: int = 2) -> Union[int, float]:
    try:
        num = float(num_str)
        if num.is_integer():
            return int(num)
        else:
            return round(num, round_num)
    except ValueError:
        raise ValueError(f"convert type error: '{num_str}'")


def _clean_svg_desc_output(svg_string: str):
    # 1. Remove Spaces between labels
    svg_string = re.sub(r'\s*\[\s*', '[', svg_string)
    svg_string = re.sub(r'\s*\]\s*', ']', svg_string)

    # 2. Leave Spaces between numbers and coordinates
    svg_string = re.sub(r'(\d)\s+', r'\1 ', svg_string)

    # 3. Remove Spaces between numbers (non-coordinate Spaces)
    svg_string = re.sub(r'\]\s*\[', '][', svg_string)

    return svg_string


def _to_svg_description(input_string: str):
    # Combine all mappers into one dictionary
    combined_mapper = {**PathMapper, **PathCMDMapper, **ShapeMapper, **ContainerMapper, **ShapeMapper,
                       **GradientsMapper, **AttribMapper}

    # This regex will match either a key from the mappers (e.g., [m], _fill_, etc.) or any other word (coordinates, colors)
    pattern = re.compile('|'.join(map(re.escape, combined_mapper.keys())))

    # Function to replace using the combined_mapper
    def replacement(match):
        key = match.group(0)
        if key in combined_mapper:
            return combined_mapper[key]  # Replace mapped keys without spaces
        return key  # Keep other parts unchanged

    # Use regex sub to apply the replacement function
    result = pattern.sub(replacement, input_string)

    # Remove spaces between brackets but preserve them between numbers or non-bracket characters
    result = re.sub(r'\]\s+\[', '][', result)

    return result


def _gather_path_attr(path_attributes: str):
    attr_ret = ""
    if path_attributes.get('fill', None):  # filled color
        attr_ret += f" _fill_{path_attributes['fill']}"
    if path_attributes.get('stroke', None):
        attr_ret += f" _stroke_{path_attributes['stroke']}"  # stroke color
    if path_attributes.get('stroke-linecap', None):
        attr_ret += f" _stroke-linecap_{path_attributes['stroke-linecap']}"
    if path_attributes.get('stroke-width', None):
        attr_ret += f" _stroke-width_{_parse_number(path_attributes['stroke-width'])}"
    if path_attributes.get('opacity', None):
        attr_ret += f" _opacity_{_parse_number(path_attributes['opacity'])}"
    return attr_ret


def _parse_point_attr(d_string: str, float_coords: bool = False):
    path_d_ret = ""
    path_command_pattern = re.compile(r'([a-zA-Z])\s*([-0-9.,\s]*)')
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')  # matching decimal, negative, scientific notation
    matches = path_command_pattern.findall(d_string)

    for _, points in matches:
        point_values = coord_pattern.findall(points)

        # coords string to float type
        if float_coords:
            float_values = [_parse_number(p) for p in point_values if p]
            points = ' '.join(map(str, float_values))
        else:
            points = ' '.join(point_values)

        path_d_ret += f"{points} "

    return path_d_ret


def _parse_path_d(d_string: str, indent_level: int = 0, float_coords: bool = False):
    path_d_ret = ""
    path_command_pattern = re.compile(r'([a-zA-Z])\s*([-0-9.,\s]*)')
    coord_pattern = re.compile(r'[-+]?\d*\.?\d+(?:e[-+]?\d+)?')  # matching decimal, negative, scientific notation
    matches = path_command_pattern.findall(d_string)

    for command, points in matches:
        point_values = coord_pattern.findall(points)

        # coords string to float type
        if float_coords:
            # float_values = [float(p) for p in point_values if p]
            float_values = [_parse_number(p) for p in point_values if p]
            points = ' '.join(map(str, float_values))
        else:
            points = ' '.join(point_values)

        if command == 'z':
            path_d_ret += f"[{command}]"
            # path_d_ret += "  " * (indent_level + 1) + f"C: {command}(closed-form) \n"
        else:
            path_d_ret += f"[{command}]{points} "
            # path_d_ret += "  " * (indent_level + 1) + f"C: {command}, coords: {points} \n"

    return path_d_ret


def _extract_elements_and_attributes(svg_description):
    # Matches the command in square brackets and the attributes that follow
    pattern = r"\[([^\]]+)\](.*?)(?=\[|$)"

    matches = re.findall(pattern, svg_description)

    # list: [(command_1, attribute_1) ... (command_N, attribute_N)]
    result = [(command.strip(), attributes.strip()) for command, attributes in matches]

    return result


def list_contains(list1, list2):
    return set(list2).issubset(set(list1))


def parse_svg_description(svg_description):
    """
    Converts a tokenized SVG description back to valid SVG markup.

    This function parses the SVG description tokens and reconstructs the corresponding
    SVG elements, handling paths, shapes, text elements, groups, and gradients.

    Args:
        svg_description (str): A string containing the tokenized SVG description
            generated by the svg2syntactic function.

    Returns:
        str: A valid SVG markup string.
    """
    # Predefine SVG Header
    svg_header = '<svg height="128" width="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">'

    svg_elements = []
    stack = []  # Tracking current open element tags
    path_data = []
    attributes = {}
    text_content = None
    g_stack = []  # For tracking nested <g> tags and their attributes

    # Store gradient definitions
    gradient_defs = {}
    current_gradient_id = None
    current_gradient_type = None

    COMMON_ATTRS = ['fill', 'opacity', 'stroke', 'stroke-width', 'stroke-linecap']

    tokens = svg_description.split('[<|')
    for i, token in enumerate(tokens):
        if '|>]' in token:
            tag, content = token.split('|>]', 1)
            tag, content = tag.strip(), content.strip()  # tag name without []

            # Handle SVG container tags
            if tag in remove_square_brackets(SVGToken['start']):
                svg_elements.append(svg_header)
            elif tag in remove_square_brackets(SVGToken['end']):
                svg_elements.append('</svg>')
            elif tag == 'start_of_g':
                stack.append('g')
                g_stack.append({})  # Create new attribute dictionary for g tag
                svg_elements.append('<g>')
            elif tag == 'end_of_g':
                if stack and stack[-1] == 'g':
                    stack.pop()
                    if g_stack:
                        g_stack.pop()
                    svg_elements.append('</g>')

            # Handle gradient-related tags
            elif tag == 'svg_linearGradient':
                current_gradient_type = 'linearGradient'
                attributes = {}
                stack.append(current_gradient_type)
            elif tag == 'svg_radialGradient':
                current_gradient_type = 'radialGradient'
                attributes = {}
                stack.append(current_gradient_type)
            elif tag == 'svg_stop':
                if stack and stack[-1] in ['linearGradient', 'radialGradient']:
                    stop_attrs = {}
                    svg_elements.append(f'<stop {to_attr(stop_attrs)} />')

            # Handle path element
            elif tag == 'svg_path':
                path_data = []
                attributes = {}
                stack.append('path')

            # Handle basic shape elements
            elif tag in ShapeIdentifier.keys():
                stack.append(ShapeIdentifier[f"{tag}"])
                attributes = {}
                if ShapeIdentifier[f"{tag}"] == 'text':
                    text_content = None

            # Handle attributes and content
            elif stack:
                current_element = stack[-1]

                # Path data processing
                if current_element == 'path':
                    if tag == 'path_d':
                        path_data.append(content)
                    elif tag in PathCMDIdentifier.keys():
                        path_data.append(f"{PathCMDIdentifier[tag]}{content}")
                    elif tag in COMMON_ATTRS:
                        attributes[tag.replace('_', '')] = content

                    # Generate path element
                    if (i + 1) < len(tokens) and is_next_svg_tag(tokens, i):
                        # Merge g tag attributes if applicable
                        if g_stack:
                            for g_attrs in g_stack:
                                for k, v in g_attrs.items():
                                    if k not in attributes:
                                        attributes[k] = v

                        svg_elements.append(f'<path d="{"".join(path_data)}" {to_attr(attributes)} />')
                        path_data = []
                        stack.pop()

                # Text element processing
                elif current_element == 'text':
                    if tag == 'text-content':
                        text_content = content
                    else:
                        attributes[tag.replace('_', '')] = content

                    if (i + 1) < len(tokens) and is_next_svg_tag(tokens, i):
                        # Handle all attributes for text element
                        text_attrs = to_attr(attributes)
                        svg_elements.append(f'<text {text_attrs}>{text_content or ""}</text>')
                        stack.pop()

                # Ellipse processing
                elif current_element == 'ellipse':
                    if tag in {'cx', 'cy', 'rx', 'ry', *COMMON_ATTRS}:
                        attributes[tag.replace('_', '')] = content

                    requires = ['rx', 'ry']
                    if list_contains(attributes.keys(), requires) and (i + 1) < len(tokens) and is_next_svg_tag(tokens,
                                                                                                                i):
                        svg_elements.append(f'<ellipse {to_attr(attributes)} />')
                        stack.pop()

                # Rectangle processing
                elif current_element == 'rect':
                    if tag in ['width', 'height', 'x', 'y', 'rx', 'ry', *COMMON_ATTRS]:
                        attributes[tag.replace('_', '')] = content

                    requires = ['width', 'height']
                    if list_contains(attributes.keys(), requires) and (i + 1) < len(tokens) and is_next_svg_tag(tokens,
                                                                                                                i):
                        svg_elements.append(f'<rect {to_attr(attributes)} />')
                        stack.pop()

                # Circle processing
                elif current_element == 'circle':
                    if tag in ['r', 'cx', 'cy', *COMMON_ATTRS]:
                        attributes[tag.replace('_', '')] = content

                    requires = ['r']
                    if list_contains(attributes.keys(), requires) and (i + 1) < len(tokens) and is_next_svg_tag(tokens,
                                                                                                                i):
                        svg_elements.append(f'<circle {to_attr(attributes)} />')
                        stack.pop()

                # Line processing
                elif current_element == 'line':
                    if tag in {'x1', 'y1', 'x2', 'y2', *COMMON_ATTRS}:
                        attributes[tag.replace('_', '')] = content

                    if (i + 1) < len(tokens) and is_next_svg_tag(tokens, i):
                        svg_elements.append(f'<line {to_attr(attributes)} />')
                        stack.pop()

                # Polygon processing
                elif current_element == 'polygon':
                    if tag in {'points', *COMMON_ATTRS}:
                        attributes[tag.replace('_', '')] = content

                    requires = ['points']
                    if list_contains(attributes.keys(), requires) and (i + 1) < len(tokens) and is_next_svg_tag(tokens,
                                                                                                                i):
                        svg_elements.append(f'<polygon {to_attr(attributes)} />')
                        stack.pop()

                # Polyline processing
                elif current_element == 'polyline':
                    if tag in {'points', *COMMON_ATTRS}:
                        attributes[tag.replace('_', '')] = content

                    requires = ['points']
                    if list_contains(attributes.keys(), requires) and (i + 1) < len(tokens) and is_next_svg_tag(tokens,
                                                                                                                i):
                        svg_elements.append(f'<polyline {to_attr(attributes)} />')
                        stack.pop()

                # Group tag attribute processing
                elif current_element == 'g':
                    if g_stack:
                        g_stack[-1][tag.replace('_', '')] = content

                # Gradient processing
                elif current_element in ['linearGradient', 'radialGradient']:
                    if tag == 'id':
                        current_gradient_id = content
                        attributes['id'] = content
                    else:
                        attributes[tag.replace('_', '')] = content

                    if (i + 1) < len(tokens) and (is_next_svg_tag(tokens, i) or 'svg_stop' in tokens[i + 1]):
                        if current_gradient_id:
                            gradient_start = f'<{current_element} {to_attr(attributes)}>'
                            gradient_defs[current_gradient_id] = {
                                'type': current_element,
                                'attrs': attributes,
                                'stops': []
                            }
                            svg_elements.append(gradient_start)

    # Handle any unclosed tags
    for element in reversed(stack):
        if element in ['linearGradient', 'radialGradient']:
            svg_elements.append(f'</{element}>')
        elif element == 'g':
            svg_elements.append('</g>')

    return ''.join(svg_elements)


def to_attr(attributes):
    """
    Converts an attribute dictionary to an SVG attribute string.

    Args:
        attributes (dict): Dictionary of attribute names and values.

    Returns:
        str: Formatted string of SVG attributes.
    """
    # Filter out None values and special attributes
    filtered_attrs = {
        k: v for k, v in attributes.items()
        if v is not None and k != 'text-content' and not k.startswith('_')
    }
    return " ".join(f"{k.replace('_', '-')}='{v}'" for k, v in filtered_attrs.items())


def is_next_svg_tag(tokens, i):
    """
    Checks if the next token is the start of an SVG tag.

    Args:
        tokens (list): List of tokens from the SVG description.
        i (int): Current token index.

    Returns:
        bool: True if the next token is an SVG tag, False otherwise.
    """
    if i + 1 >= len(tokens):
        return True  # If this is the last token, consider the next one as an end tag

    next_token = tokens[i + 1]
    svg_tag_identifiers = set(list(ShapeIdentifier.keys()) +
                              [PathIdentifier] +
                              list(GradientIdentifier.values()) +
                              list(ContainerTagIdentifiers.values()))

    return any(identifier in next_token for identifier in svg_tag_identifiers)


def syntactic2svg(svg_description: str, print_info: bool = False) -> str:
    """
    Converts a tokenized SVG description back to a valid SVG markup string.

    This function parses the syntactic representation generated by svg2syntactic
    and reconstructs the corresponding SVG XML. It handles path data, shape elements,
    text elements, groups, and basic attributes.

    Args:
        svg_description (str): A string containing the tokenized SVG description.
        print_info (bool, optional): If True, prints debugging information. Defaults to False.

    Returns:
        str: A valid SVG markup string.
    """
    if print_info:
        elements_and_attributes = _extract_elements_and_attributes(svg_description)
        print(elements_and_attributes)

    # Parse the commands and generate SVG
    svg_output = parse_svg_description(svg_description)

    return svg_output


def syntactic2svg(svg_description: str, print_info: float = False) -> str:
    if print_info:
        elements_and_attributes = _extract_elements_and_attributes(svg_description)
        print(elements_and_attributes)

    # Parse the commands
    svg_output = parse_svg_description(svg_description)

    return svg_output


if __name__ == '__main__':
    # path and circle
    string_1f408_200d_2b1b_black_cat = '<svg height="128" viewBox="0 0 128 128" width="128" xmlns="http://www.w3.org/2000/svg"><path d="m37.26 79.78s2.49 8.11-1.2 28.42c-.66 3.65-1.64 7.37-2.13 10.42-6.25 0-6.58 7.12-5.26 7.12h7.45c4.75 0 10.56-11.86 13.7-28.31s-12.56-17.65-12.56-17.65zm46.37 13.17s8.07 4.33 7.78 14.51c-.12 4.02-.89 7.43-1.27 10.75-6.51 0-6.53 7.06-4.64 7.06h6.6c3.28 0 10.67-11.23 12.02-25.89 1.34-14.66-20.49-6.43-20.49-6.43z" fill="#292f33"/><path d="m128 30.03c0-17.5-14.72-26.47-32-26.47-1.97 0-3.56 1.59-3.56 3.55s1.59 3.56 3.56 3.56c6.41 0 23.88 3.32 23.88 19.36 0 10.25-5.57 19.42-13.4 21.27-.58-.26-1.23-.37-1.81-.68-25.74-13.93-47.85 3.21-54.14-1.03-6.23-4.2-1.91-11.57-10.04-18.64-1.17-6.55-3.81-15.92-6.38-15.92-1.95 0-4.5 6.49-6.18 13-2.11-4.91-4.85-9.8-6.62-9.8-2.27 0-4.54 8.08-5.67 15.06-5.97 4.08-10.33 9.98-10.33 16.94 0 9.6 14.4 11.94 19.2 12.13s10.75 12.66 12.75 18.03c4.41 15.53 7.29 30.93 9.59 39.76-5.86 0-6.41 7.51-4.79 7.51 2.53 0 6.94-.01 7.91 0 4.91.05 7.2-16.73 7.2-31.46 0-.76-.04-2.23-.04-2.23s6.87 1.79 21.47-.74c8.69-1.51 17.89 3.02 20.43 11.24 1.88 6.06 4.98 11.75 6.64 15.95-5.65 0-5.49 7.24-3.85 7.24 2.8 0 6.4.05 7.76 0 5.22-.2 2.29-26.93 3.66-35.9 1.38-8.97 4.51-19.83-.81-31.28 11.51-6.31 15.57-19.57 15.57-30.45" fill="#292f33"/><circle cx="21.312" cy="41.841778" fill="#c3c914" r="3.2"/><path d="m10.61 45.72c-2.41 1.51-2.41 6.32-3.61 6.32s-3.61-2.83-3.61-6.32c0-3.48 10.18-1.84 7.22 0m12.78 5.28c-.15.09-.32.13-.51.09-.06-.01-5.5-.86-9.05.52-.45.18-1.21-.08-1.46-.68-.26-.6.08-1.3.54-1.48 5.12-2.02 10.15-.83 10.43-.77.49.1.69.67.67 1.32-.02.41-.37.84-.62 1m-1.68 8.31c-.17.04-.34 0-.5-.11-.05-.03-4.73-2.95-8.54-3.03-.49-.01-1.09-.55-1.09-1.21s.59-1.19 1.08-1.19c5.52.09 9.69 3.17 9.92 3.34.42.29.37.9.1 1.5-.18.38-.67.65-.97.7z" fill="#66757f"/><path d="m27.93 28.02s1.32-.14 3.05.12c.77-1.64 2.08-3.16 2.08-3.16s1 2.38 1.37 4.2c.09.42.78.41 1.15.67 0 0 .13-12.31-3.07-13.64 0-.01-2.69 4-4.58 11.81zm-12.29 5.26s1.37-.95 3.14-2.04c.77-2.22 2.02-4.74 2.02-4.74s1.09 2.14 1.23 2.88c.74-.41.74-.38 1.43-.68 0 0-.41-8.12-3.91-8.98 0 0-2.6 4.88-3.91 13.56z" fill="#7f676d"/><path d="m65.6 84.07c-9.42 2.02-8.46 9.89-8.46 9.89s6.86 1.79 21.47-.74c3.79-.66 7.65-.11 11.04 1.34-5.71-10.83-15.34-12.37-24.05-10.49" fill="#66757f"/><path d="m21.25 39.05s1.18 1.23 1.13 2.76-.89 2.97-.89 2.97-1.03-1.24-1.13-2.94c-.1-1.71.89-2.79.89-2.79" fill="#292f33"/></svg>'
    ret1, ret2 = svg2syntactic(string_1f408_200d_2b1b_black_cat, include_gradient_tag=False, include_group=True)

    # emoji_u1f3ca_1f3ff_200d_2640_clean.svg
    # emoji_u1f3ca_1f3ff_200d_2640_clean_man_swim = '<svg enable-background="new 0 0 128 128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><linearGradient id="a" gradientTransform="matrix(1 0 0 -1 0 128)" gradientUnits="userSpaceOnUse" x1="47.1165" x2="47.1165" y1="95.4031" y2="54.6588"><stop offset="0" stop-color="#70534a"/><stop offset="1" stop-color="#5c4037"/></linearGradient><linearGradient id="b"><stop offset="0" stop-color="#651fff"/><stop offset=".7047" stop-color="#5914f2"/><stop offset="1" stop-color="#530eeb"/></linearGradient><linearGradient id="c" gradientTransform="matrix(1 0 0 -1 0 128)" gradientUnits="userSpaceOnUse" x1="46.721" x2="113.15" xlink:href="#b" y1="31.972" y2="31.972"/><clipPath id="d"><ellipse cx="37.05" cy="80.86" rx="3.94" ry="2.95" transform="matrix(.5624 -.8269 .8269 .5624 -50.6489 66.0249)"/></clipPath><clipPath id="e"><ellipse cx="45.28" cy="68.69" rx="3.94" ry="2.95" transform="matrix(.5624 -.8269 .8269 .5624 -36.9824 67.4982)"/></clipPath><linearGradient id="f" gradientTransform="matrix(1 0 0 -1 0 128)" gradientUnits="userSpaceOnUse" x1="21.0165" x2="34.3917" xlink:href="#b" y1="65.0851" y2="57.6833"/><path d="m21.18 42.61 11.72-2.09c1.57-.35 2.57-1.9 2.22-3.47-.35-1.57-1.9-2.57-3.47-2.22l-11.61 3.23c-1.57.35-2.63 1.38-2.29 2.95.35 1.57 1.86 1.95 3.43 1.6z" fill="#4a2f27"/><path d="m89.53 70.79c-5.11-2.41-9.65-6.04-13.14-10.5l-16.75-21.44c-1.01-1.66-2.6-2.81-4.49-3.26-.08-.02-.16-.03-.24-.05l.01-.07-34.28-2.99c-1.07-.09-2.14.02-3.17.34l-5.32 1.66c-.7.22-1.37.53-2 .93-1.24.79-3.47 2.26-4.5 3.16-.57.5-.95 1.01-.95 1.92 0 .74.3 1.4.86 1.86.61.51 1.47.71 2.35.55 1.11-.2 2.34-1.27 3.18-2.13.23-.24.53-.39.86-.43l5.63-.74c.43-.06.85-.04 1.25.05.75.17 1.63.32 2.66.51 1.33.24 2.99.53 4.93.96 2.4.53 20.25 6.29 20.42 6.34 1.21.42 2.27 1.23 2.98 2.3l6.72 10.1c2.53 3.8 2.5 8.81-.06 12.47l-.1.14c-2.71 3.91-5.94 10.83-7.19 15.43-2.69 9.87-1.06 16.29 4.87 19.09l22.85 10.79z" fill="url(#a)"/><path d="m63.04 113.5 19.74 9.35 2.34 1.1h9.24l18.79-39.64-28.35-16.21c-7.7 8.28-25.37 1.56-26.76.88-.25.6-.61 1.13-.97 1.75 11.19 8.98 7.71 16.51 7.71 16.51-.03.06-3.67 7.47-17.64 4.52-.16.69-.3 1.38-.41 2.05 1.4.66 17.83 8.48 16.31 19.69z" fill="url(#c)"/><path d="m61.14 87.92c1.89-2.85 1.11-6.69-1.74-8.58l-4.41-2.92-6.83 10.32 4.41 2.92c2.84 1.89 6.68 1.11 8.57-1.74z" fill="#3c2b24"/><path d="m19.95 60.96s.13-.21.19-.3.2-.3.2-.3c8-10.58 15.78-8.85 20.52-6.69 5.14 2.33 8.71 4.42 8.71 4.42s-.31.06-.81.22c-.52.18-.95.47-.95.47l3.54 3.19-10.64-1.48c-.07-.01-.13.02-.16.08l-1.26 2.46c-1.27 2.47-2.68 4.87-4.2 7.19h.01c-.07.1-.13.2-.2.3s-.13.2-.2.3h-.01c-1.54 2.31-3.2 4.54-4.98 6.68l-1.77 2.12c-.04.05-.05.12-.01.18l5.52 9.22-4.32-2.02s-.11.51-.07 1.06c.04.52.11.83.11.83s-3.68-2.35-7.62-6.39c-3.73-3.83-8.05-10.36-1.6-21.54z" fill="#6d4c41"/><path d="m52.1 58.71c-2.69-1.78-4.77.43-4.77.43l-.76 1.15-16.51 24.93-.76 1.15s-1.22 2.78 1.47 4.56c2.37 1.57 4.51.37 5.72-1.45l16.51-24.94c1.2-1.82 1.47-4.26-.9-5.83z" fill="#3c2b24"/><path d="m20.71 61.1c7.12-10.76 17.33-8.05 29.04-.29 12.15 8.04 10.12 17.93 6.3 23.7-3.76 5.68-12.04 11.55-24.27 3.45-11.72-7.77-18.19-16.1-11.07-26.86z" fill="#70534a"/><path d="m44.07 62.65s-2.06-.05-3.49 2.08-.6 4.02-.6 4.02c.05.14.15.28.29.37.35.23.82.14 1.06-.21.05-.07.12-.29.12-.31.33-1.63 1-2.63 1-2.63s.67-.99 2.05-1.91c.02-.01.19-.15.24-.22.23-.35.14-.82-.21-1.06-.15-.1-.31-.14-.46-.13z" fill="#1a1717"/><path d="m35.18 75.91s-2.06-.05-3.49 2.08-.6 4.02-.6 4.02c.05.14.15.28.29.37.35.23.82.14 1.06-.21.05-.07.12-.29.12-.31.33-1.63 1-2.63 1-2.63s.67-.99 2.05-1.91c.02-.01.19-.15.24-.22.23-.35.14-.82-.21-1.06-.15-.1-.31-.14-.46-.13z" fill="#1a1717"/><path d="m40.21 76.18c1.84 1.24 2.3 3.78.17 6.93-2.1 3.11-4.66 3.68-6.5 2.44s-2.77-3.92-.56-7.18c2.14-3.18 5.05-3.43 6.89-2.19z" fill="#01579b"/><ellipse cx="37.05" cy="80.86" fill="#b3e5fc" rx="3.94" ry="2.95" transform="matrix(.5624 -.8269 .8269 .5624 -50.6489 66.0249)"/><path clip-path="url(#d)" d="m39.97 85.71-4.73-8.11.58-3.06 4.73 8.11z" fill="#ffffff"/><path d="m48.44 64.01c1.84 1.24 2.3 3.78.17 6.93-2.1 3.11-4.66 3.68-6.5 2.44s-2.87-3.87-.67-7.13c2.15-3.18 5.16-3.48 7-2.24z" fill="#01579b"/><ellipse cx="45.28" cy="68.69" fill="#b3e5fc" rx="3.94" ry="2.95" transform="matrix(.5624 -.8269 .8269 .5624 -36.9824 67.4982)"/><path clip-path="url(#e)" d="m48.2 73.54-4.73-8.11.58-3.07 4.73 8.11z" fill="#ffffff"/><g fill="#01579b"><path d="m43.49 73.6-2.55 3.77-1.88-1.27 2.55-3.77z"/><path d="m48.23 59.86c1.26 1.5 1.36 3.38.96 5.3l-1.88-1.26c.71-1.56.57-3.33-.96-5.3z"/><path d="m29.96 86.91c1.87.6 3.65-.01 5.28-1.11l-1.88-1.26c-1.18 1.25-2.88 1.78-5.28 1.11z"/></g><path d="m18.1 59.1s-.15.21-.21.3-.19.3-.19.3c-7.01 11.69-1.42 20 2.69 23.45 4.56 3.83 8.65 5.67 8.65 5.67s-.08-.32-.13-.84c-.06-.56.04-1.07.04-1.07s-1.44-5.54-.72-10.09c.74-4.65 5.79-12.21 9.3-14.43 3.72-2.35 5.87-3.38 10.24-3.63 0 0 .44-.29.98-.46.51-.16.83-.21.83-.21s-3.4-2.83-8.64-5.68c-3.98-2.16-14.33-4.43-22.84 6.69z" fill="url(#f)"/><path d="m46.61 76.09c-.04.03-.08.07-.12.11l-1.84 2.9c-.02.05-.04.1-.05.15-.06.33.12.65.49.71s1.55 0 2.29-1.17c.74-1.16.29-2.26.08-2.57-.21-.3-.58-.33-.85-.13z" fill="#33251f"/><ellipse cx="50.48" cy="80.97" fill="#1a1717" rx="3.19" ry="1.93" transform="matrix(.6002 -.7998 .7998 .6002 -44.5827 72.747)"/><path d="m112 81.98c-6.57 0-9.93 7.19-16 7.19-6.06 0-9.43-7.19-16-7.19s-9.93 7.19-16 7.19-9.43-7.19-16-7.19-9.94 7.19-16 7.19-9.43-7.19-16-7.19c-5.01 0-8.16 4.18-12.06 6.16v35.82h120v-35.88c-3.83-2-6.97-6.1-11.94-6.1z" fill="#039be5"/><path d="m111.97 105.69c-6.57 0-9.93-7.19-16-7.19-6.06 0-9.43 7.19-16 7.19s-9.93-7.19-16-7.19-9.43 7.19-16 7.19-9.94-7.19-16-7.19-9.43 7.19-16 7.19c-5 0-8.14-4.16-12.03-6.15v24.41h120v-24.39c-3.84 2-6.99 6.13-11.97 6.13z" fill="#29b6f6"/></svg>'
    # ret1, ret2 = svg2syntactic(emoji_u1f3ca_1f3ff_200d_2640_clean_man_swim, include_gradient=True, include_g=False)

    # emoji_u1f469_1f3fd_200d_1f9bd_200d_27a1_clean = '<svg enable-background="new 0 0 128 128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"><linearGradient id="a"><stop offset=".5264" stop-color="#6d4c41" stop-opacity="0"/><stop offset="1" stop-color="#6d4c41"/></linearGradient><radialGradient id="b" cx="-962.9219" cy="641.0479" gradientTransform="matrix(-13.1769 0 0 13.1769 -12639.5488 -8416.2314)" gradientUnits="userSpaceOnUse" r="1" xlink:href="#a"/><radialGradient id="c" cx="-951.5182" cy="634.9957" gradientTransform="matrix(-15.8356 0 0 15.8356 -15028.5332 -10014.5752)" gradientUnits="userSpaceOnUse" r="1"><stop offset="0" stop-color="#6d4c41"/><stop offset=".5264" stop-color="#6d4c41" stop-opacity="0"/></radialGradient><radialGradient id="d" cx="-952.3141" cy="635.4181" gradientTransform="matrix(-15.6157 0 0 15.6157 -14804.6172 -9899.6455)" gradientUnits="userSpaceOnUse" r="1" xlink:href="#a"/><radialGradient id="e" cx="-946.2455" cy="613.8471" gradientTransform="matrix(-18.5134 -6.0729 -6.0729 18.5134 -13729.707 -17056.4551)" gradientUnits="userSpaceOnUse" r="1"><stop offset=".3637" stop-color="#651fff"/><stop offset=".8121" stop-color="#5914f2"/><stop offset="1" stop-color="#530eeb"/></radialGradient><radialGradient id="f" cx="-1703.7103" cy="1034.204" gradientTransform="matrix(-1.1067 0 0 1.1067 -1837.9663 -1044.2474)" gradientUnits="userSpaceOnUse" r="1.0003"><stop offset="0" stop-color="#78909c"/><stop offset=".5619" stop-color="#617a86"/><stop offset="1" stop-color="#546e7a"/></radialGradient><linearGradient id="g"><stop offset=".5" stop-color="#ba8d68"/><stop offset="1" stop-color="#a47b62"/></linearGradient><linearGradient id="h" gradientTransform="matrix(1 0 0 -1 0 130)" gradientUnits="userSpaceOnUse" x1="42.0636" x2="54.0635" xlink:href="#g" y1="62.9225" y2="71.2946"/><linearGradient id="i" gradientTransform="matrix(1 0 0 -1 0 130)" gradientUnits="userSpaceOnUse" x1="53.7638" x2="52.6475" y1="48.6762" y2="51.2808"><stop offset="0" stop-color="#ba8d68"/><stop offset="1" stop-color="#a47b62"/></linearGradient><linearGradient id="j" gradientTransform="matrix(1 0 0 -1 0 130)" gradientUnits="userSpaceOnUse" x1="52.4488" x2="54.6814" xlink:href="#g" y1="52.2052" y2="48.2982"/><path d="m39.03 42.85c.03-.02 2.58-1.67 3.14-4.46.36-1.79-.18-4.06-.76-6.45-1.1-4.55-2.35-9.7 2.4-13.25 5.17-3.86 11.15-.46 11.21-.43.16.09.25.26.25.44l-.02 1.77c0 .24-.17.44-.4.49-.03.01-3.23.69-4.09 3.19-.66 1.94-.62 4.18-.58 6.56.06 3.18.12 6.46-1.58 9.2-2.22 3.58-4.83 4.4-6.76 4.4-1.62 0-2.76-.58-2.78-.59-.16-.08-.26-.24-.27-.42 0-.19.09-.35.24-.45z" fill="#543930"/><path d="m41.4 31.94c.05.19.09.38.14.57l8.72-5.87c.08-.87.23-1.71.5-2.49.86-2.5 4.06-3.19 4.09-3.19.23-.05.4-.25.4-.49l.02-1.77c0-.18-.09-.34-.25-.43-.06-.04-6.05-3.43-11.22.42-4.74 3.55-3.49 8.7-2.4 13.25z" fill="url(#b)"/><path d="m38.8 43.3c.01.18.11.34.27.42.03.01 1.16.59 2.78.59 1.93 0 4.54-.82 6.76-4.4 1.7-2.74 1.64-6.02 1.58-9.19-.02-1.42-.05-2.8.08-4.08l-8.72 5.87c.52 2.18.96 4.23.63 5.89-.57 2.79-3.12 4.44-3.14 4.46-.15.09-.24.26-.24.44z" fill="url(#c)"/><path d="m55.41 18.83c2.37-4.24 8-7.71 12.82-7.14 5.4.64 8.46 4.37 9.44 9.23.36 1.76.38 3.53.14 4.8-.03.15-.28 1.26-.28 1.6-.14 1.33 2.47 2.52 2.84 3.97.28 1.09-1.98 1.89-2.15 2.09-.85 1.02.78 5.69-5.47 6.4-2.18.25-3.54-.12-4.3-.06-1.8.15-2.8 6.11-2.8 6.11l-9.21-5.57s3.07-4.51.39-9.14c-2.59-3.68-2.94-9.56-1.42-12.29z" fill="#ba8d68"/><path d="m77.69 36.04c-.87.09-1.74-.08-2.5-.59-.83-.56-.31-1.25.63-.9.57.21 1.36.29 2.07.13z" fill="#5d4037"/><path d="m54.82 15.79c2.24-3.03 6.04-5.08 10.16-5.49 3.58-.36 8.54.28 11.72 3.12 1.91 1.7 3.1 4.02 2.94 6.66-.1 1.59-1.68 2.41-1.68 2.41l-.08-.13s-.01-.02-.03-.05l-.14-.25c-.5-.8-1.66-2.93-4.72-1.95-3.78 1.21-5.52.45-5.52.45 1.45 2.99-1.06 6.62-1.34 7.02-.15.21-.28.2-.35-.01-.17-.49-.5-1.36-.74-1.77-.78-1.37-1.9-.81-1.91-.81-1.53.62-1.92 3.84.72 5.59.06.04.21.14.16.47-.12.81-1.53 3.12-3.85 3.8-.31.09-.58.15-.83.18-1.32.15-2.08-.36-2.28-1.34-.43-2.14-.93-2.74-1.65-3.64-.83-1.04-1.78-2.22-2.45-5.57-.64-3.16.01-6.17 1.87-8.69z" fill="#543930"/><path d="m54.82 15.79c2.24-3.03 6.04-5.08 10.16-5.49 3.58-.36 8.54.28 11.72 3.12 1.91 1.7 3.1 4.02 2.94 6.66-.1 1.59-1.68 2.41-1.68 2.41l-.08-.13s-.01-.02-.03-.05l-.14-.25c-.5-.8-1.66-2.93-4.72-1.95-3.78 1.21-5.52.45-5.52.45 1.45 2.99-1.06 6.62-1.34 7.02-.15.21-.28.2-.35-.01-.17-.49-.5-1.36-.74-1.77-.78-1.37-1.9-.81-1.91-.81-1.53.62-1.92 3.84.72 5.59.06.04.21.14.16.47-.12.81-1.53 3.12-3.85 3.8-.31.09-.58.15-.83.18-1.32.15-2.08-.36-2.28-1.34-.43-2.14-.93-2.74-1.65-3.64-.83-1.04-1.78-2.22-2.45-5.57-.64-3.16.01-6.17 1.87-8.69z" fill="url(#d)"/><path d="m73.09 28.28c-.01-.98.49-1.78 1.13-1.79s1.16.78 1.17 1.75c.01.98-.49 1.78-1.13 1.79-.63.01-1.16-.77-1.17-1.75z" fill="#49362e"/><path d="m75.01 25.61c1.15.22 1.34-.55.91-1.1-.32-.41-1.03-.71-2.31-.32-1.21.36-1.64 1.12-1.93 1.59-.29.46-.21.89.08.9.39 0 1.83-1.34 3.25-1.07z" fill="#613e31"/><path d="m84.57 116.74c-.6.19-1.25-.14-1.44-.74l-.85-2.67c-.58-1.81.59-4.47.59-4.47l6.64-2.14 6.61.05c1.63.1 3.04 1.19 3.55 2.74l.39 1.2c.17.53-.12 1.09-.65 1.26z" fill="#4568ad"/><path d="m44.86 67.06 17.02 3.62.13 2.55 7.94.18c4.88.48 11.67.17 14.27 10.49l6.89 22.59-9.85 3.43s-7.9-21.46-8.4-21.43c-.5.02-15.14.81-24.77-.5-9.6-1.29-3.23-20.93-3.23-20.93z" fill="#7c7c7c"/><path d="m63.91 72.18c-1.55-12.08 1.22-12.23 3.31-20.12 1.06-4-.37-8.61-1.95-10.5-3.47-3.15-7.2-3.97-10.19-1.2-5.89 5.48-12.5 26.92-12.5 26.92 4.71 8.68 21.33 4.9 21.33 4.9z" fill="url(#e)"/><path d="m45.53 93.17h4v7.14h-4z" fill="#c62828"/><path d="m84.76 119.99 15.98-5.19c1.16-.38 1.8-1.63 1.42-2.79s-1.63-1.8-2.79-1.42l-15.98 5.19c-1.16.38-1.8 1.63-1.42 2.79s1.63 1.8 2.79 1.42z" fill="#424242"/><path d="m78.46 111.63c3.61 0 6.53 2.93 6.53 6.53 0 3.61-2.93 6.53-6.53 6.53-3.61 0-6.53-2.93-6.53-6.53-.01-3.6 2.92-6.53 6.53-6.53z" fill="#78909c"/><path d="m34.38 59.87c1.81 0 1.81 1.94 1.81 2.59l.25 29.21 32.67-1.77c9.36 0 13.66 3.58 17.04 14.18l4.78 13.87c.17.5.14 1.05-.09 1.53s-.64.84-1.15 1.01c-.21.07-.43.11-.65.11-.85 0-1.61-.54-1.89-1.35l-3.55-10.29c-1.84.6-3.14 2.35-3.14 4.32v4.88c0 1.1-.9 2-2 2s-2-.9-2-2v-4.88c0-3.68 2.38-6.94 5.84-8.1-2.84-8.85-5.68-11.28-13.07-11.29l-32.85 1.91c-1.25 0-1.92-.14-2.63-.96-.79-.95-.8-2.47-.8-2.48l-.38-30.19c0-2 1.13-2.3 1.81-2.3z" fill="#f44336"/><path d="m78.48 120.33c.99 0 1.8-.66 1.96-1.65l.33-3.35c.08-.53-.37-1-.95-1h-2.54c-.57 0-1.01.45-.96.97l.12 3.39c0 .9 1.05 1.64 2.04 1.64z" fill="#424242"/><path d="m35.03 62.84c0-2.46 1.34-2.85 3.55-2.85h1.64c2.31.03 2.69 1.07 3.11 2.57.75 2.67 2.18 16.77 1.07 20.67-.03.1-.08.22-.12.32l28.96-.85c1.73 0 3.19.63 3.77 4.09 0 0 .11.44-.09 2.12s-1.31 2.44-2.76 2.51l-38.84 1.88.05-.13c-.06.05-.1.08-.1.08z" fill="#424242"/><path d="m57.21 92.15c-2.22-2.9-5.67-4.56-10.02-4.35-5.04.23-8.15 3.31-9.74 5.4l-3.4.07c-.07-3.89-.26-13.9-.26-13.9 2.54-4.26 29.74-11.62 38.87 12.08z" fill="#212121"/><path d="m35.34 93.95-1.81-.56c0-2.09.26-15.91.26-15.91 2.67-4.46 31.97-12.63 40.07 15.72l-5.24-.02-8.37.49c-2.02-4.53-7.42-7.66-13.19-7.4-8.58.4-11.72 7.68-11.72 7.68z" fill="#f44336"/><path d="m47.53 101.41c.61 0 1.11-.5 1.11-1.11s-.5-1.11-1.11-1.11-1.11.5-1.11 1.11.5 1.11 1.11 1.11z" fill="url(#f)"/><path d="m47.53 99.7c-.33 0-.61.27-.61.61 0 .33.27.61.61.61s.61-.27.61-.61-.28-.61-.61-.61zm0-1c.89 0 1.61.72 1.61 1.61s-.72 1.61-1.61 1.61-1.61-.72-1.61-1.61.72-1.61 1.61-1.61z" fill="#82aec0"/><g stroke="#82aec0" stroke-linecap="round" stroke-miterlimit="10"><path d="m47.53 79.6v41.42"/><path d="m32.89 85.66 29.28 29.29"/><path d="m32.89 114.95 29.28-29.29"/><path d="m68.24 100.31h-41.42"/></g><path d="m47.53 75.91c13.47 0 24.39 10.92 24.39 24.39s-10.92 24.4-24.39 24.4-24.39-10.92-24.39-24.39 10.92-24.4 24.39-24.4zm0 44.14c10.9 0 19.74-8.84 19.74-19.74s-8.84-19.74-19.74-19.74-19.74 8.84-19.74 19.74 8.84 19.74 19.74 19.74z" fill="#78909c"/><path d="m47.53 83.58c-9.22 0-16.73 7.5-16.73 16.73s7.5 16.73 16.73 16.73 16.73-7.5 16.73-16.73-7.51-16.73-16.73-16.73zm0-1c9.79 0 17.73 7.94 17.73 17.73s-7.94 17.73-17.73 17.73-17.73-7.94-17.73-17.73 7.94-17.73 17.73-17.73z" fill="#546e7a"/><path d="m47.53 104.42c-2.27 0-4.11-1.84-4.11-4.11s1.84-4.11 4.11-4.11 4.11 1.84 4.11 4.11-1.84 4.11-4.11 4.11z" fill="#757575"/><path d="m100.18 110.93-9.45 3.04" stroke="#757575" stroke-linecap="round" stroke-miterlimit="10"/><path d="m35.14 76.21s-1.49.66-2.35 1.81l-.07-3.51c.96-.96 2.39-1.17 2.39-1.17z" fill="#c62828"/><path d="m73.3 91.41.69 2.95s1.28.12 2.23-.66c1.64-1.35 1.45-2.12 1.45-2.12-.45-.35-1.73-.78-1.73-.78s-.72.35-1.38.49-1.26.12-1.26.12z" fill="#c62828"/><path d="m68.46 93.59c-1.51-4.89-4.89-9.53-9.79-12.21" opacity=".8" stroke="#94d1e0" stroke-linecap="round" stroke-miterlimit="10" stroke-width="2"/><path d="m53.15 47.14 5.16 5.54-9.88 6.89 3.68 13.42-4.41 1.78s-5.14-11.34-6.1-13.23c-1.58-3.14-1.44-5.08.53-7.32 1.05-1.18 11.02-7.08 11.02-7.08z" fill="url(#h)"/><path d="m47.37 73.49c.23-.18.43-.35.62-.51.39-.33.76-.64 1.35-1.01.72-.44 4.15.93 4.93 1.22.26.1.5.19.72.29.3.13.53.36.63.63.04.05.08.11.12.2.01.04.02.07.02.11 0 .01.25 1.5.65 2.31.16.31.07.84-.26.96 0 0 .19.77.33 1.16.17.46.12.61.14.82.04.36-.1.63-.36.8-.32.21-.44.08-.59.19s-.1.29-.14.5c-.11.56-.58.71-.77.73-.62.06-.8-.33-.91-.33-.2.01-.24.18-.26.31-.06.28-.36.43-.64.52-.73.08-1.26-.54-1.28-.44-.08.37-.72.65-1.09.5-.31-.13-.5.12-2.35-2.72-.37-.5-.97-1.84-1.37-3.12l-.19-.61c-.3-.96-.03-1.94.7-2.51z" fill="url(#i)"/><path d="m49.06 71.88c1.76-.67 6.6 1.02 6.72 2.61.03.36-.46.92-.9.73 0 0 0 .43-.03.53-.14.54-.76.57-.95.6-.35.05-.52.13-.53.48-.01.33-.23.7-.52.87-.32.18-.57.22-.96.12-.18-.05-.28.06-.34.24-.08.25-.28.59-.49.69-.39.2-.64.16-.92.15-.18 0-.26.02-.38.16-.09.12-.08.59-.39.76-1.39.73-2.2-2.24-2.73-3.97-.39-1.26.43-3.21 2.42-3.97z" fill="url(#j)"/><path d="m53.08 57.53-5.87-9.39 8.35-5.22c2.59-1.62 6.01-.83 7.63 1.76s.83 6.01-1.76 7.63z" fill="#3615af"/></svg>'
    # ret1, ret2 = svg2syntactic(emoji_u1f469_1f3fd_200d_1f9bd_200d_27a1_clean, include_gradient=True, include_g=False)

    print(ret1)
    print(ret2)

    ret = syntactic2svg(ret2)
    print(ret)

    # svg_struct = "[<|START_OF_SVG|>][<|svg_path|>][<|d|>][<|moveto|>]454.94 54.29[<|curveto|>]-14.13 -14.23 -33.33 -22.28 -53.41 -22.28[<|horizontal_lineto|>]-7.88 -129.34 -16.63 -129.34 -7.88[<|curveto|>]-20.08 0 -39.28 8.05 -53.41 22.28 -45.33 52.92 -57.05 119.63 -57.05 201.08 0 91 36.61 171.34 106.02 219.69 59.5 30.23 136.17 -86.23 31.83 -140.94 -38.86 -20.36 -40.67 -43.28 -40.67 -76.31[<|smooth_curveto|>]26.77 -59.83 59.82 -59.83[<|horizontal_lineto|>]90.69 16.63 90.68[<|curveto|>]33.05 0 59.83 26.8 59.83 59.83[<|smooth_curveto|>]-1.81 55.95 -40.67 76.31[<|curveto|>]-104.34 54.7 -27.67 171.17 31.83 140.94 69.4 -48.35 106.01 -128.69 106.01 -219.69 0 -81.45 -11.72 -148.16 -57.06 -201.08[<|close_the_path|>][<|moveto|>]-114.91 71.35[<|curveto|>]0 6.91 -5.61 12.53 -12.53 12.53[<|horizontal_lineto|>]-143[<|curveto|>]-6.92 0 -12.53 -5.63 -12.53 -12.53[<|vertical_lineto|>]-28.02[<|curveto|>]0 -6.92 5.61 -12.53 12.53 -12.53[<|horizontal_lineto|>]143[<|curveto|>]6.92 0 12.53 5.61 12.53 12.53[<|close_the_path|>][<|END_OF_SVG|>]"
    #
    # ret = syntactic2svg(svg_struct)
    #
    # print(ret)
