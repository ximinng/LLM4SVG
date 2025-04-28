# -*- coding: utf-8 -*-
# Author: ximing
# Copyright (c) 2024, XiMing Xing.
# License: MIT License
# Description: SVG process utils

import re
import xml.etree.ElementTree as ET

from lxml import etree

"""Flatten and inherit group"""


def apply_g_attributes_to_children(svg_string):
    root = etree.fromstring(svg_string)

    for g_tag in root.xpath('//svg:g', namespaces={'svg': 'http://www.w3.org/2000/svg'}):
        # get <g> parent
        parent = g_tag.getparent()
        # <g> index
        g_index = parent.index(g_tag)
        # get <g> attributes
        g_attributes = g_tag.attrib

        for i, child in enumerate(g_tag):
            # inherit <g> attributes
            for attr, value in g_attributes.items():
                if attr not in child.attrib:
                    child.attrib[attr] = value

            # moves the <g>' child to the location of its parent
            parent.insert(g_index + i, child)

        # delete <g>
        parent.remove(g_tag)

    return etree.tostring(root, encoding="utf-8").decode()


"""Simplify gradient tags"""


def hex_to_rgb(hex_color):
    # Ensure it's a valid hex color, with optional '#'
    hex_pattern = re.compile(r'^#?([A-Fa-f0-9]{6})$')

    match = hex_pattern.match(hex_color)
    if not match:
        raise ValueError(f"Invalid hex color: {hex_color}")

    # Remove the '#' if present
    hex_color = match.group(1)

    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return '#' + ''.join(f'{c:02x}' for c in rgb)


def average_color(colors):
    if not colors:
        return None

    avg_r = sum(c[0] for c in colors) // len(colors)
    avg_g = sum(c[1] for c in colors) // len(colors)
    avg_b = sum(c[2] for c in colors) // len(colors)

    return rgb_to_hex((avg_r, avg_g, avg_b))


def get_gradient_color(gradient_id, gradients, root, ns):
    """
    find color by gradient_id，
    and Supports access to other gradients via xlink:href
    """
    if gradient_id in gradients:
        return gradients[gradient_id]

    # Find the gradient associated with the gradient id
    gradient = root.xpath(f'//svg:radialGradient[@id="{gradient_id}"] | //svg:linearGradient[@id="{gradient_id}"]',
                          namespaces=ns)
    if gradient:
        gradient = gradient[0]
        # check xlink:href
        xlink_href = gradient.get('{http://www.w3.org/1999/xlink}href')
        if xlink_href:
            referenced_id = xlink_href[1:]  # remove '#'
            return get_gradient_color(referenced_id, gradients, root, ns)

    return None  # color not found


def get_previous_fill_color(paths, index):
    """Gets the fill color of the last valid path."""
    colors = []
    while index >= 0:
        fill = paths[index].get('fill')
        if fill and not fill.startswith('url(#'):
            try:
                # Attempt to convert the color to RGB to verify it's valid
                hex_to_rgb(fill)
                colors.append(fill)
            except ValueError:
                # Skip invalid hex colors
                pass
        index -= 1
    return colors


def replace_gradient_tags(
        svg_string: str,
        fill_is_empty: str = "previous"  # 'skip', 'default'
):
    root = etree.fromstring(svg_string)

    ns = {
        'svg': 'http://www.w3.org/2000/svg',
        'xlink': 'http://www.w3.org/1999/xlink'
    }

    # find all gradient
    gradients = {}
    radial_gradients = root.findall('.//svg:radialGradient', ns)
    linear_gradients = root.findall('.//svg:linearGradient', ns)

    # extract first stop-color in gradient
    for gradient in radial_gradients + linear_gradients:
        gradient_id = gradient.get('id')
        if gradient_id is not None:
            first_stop = gradient.xpath('.//svg:stop[1]/@stop-color', namespaces=ns)
            if first_stop:
                gradients[gradient_id] = first_stop[0]
            else:
                # no stop, then access xlink:href
                color = get_gradient_color(gradient_id, gradients, root, ns)
                if color:
                    gradients[gradient_id] = color

    # get all paths
    paths = root.findall('.//svg:path', ns)

    # Replace the fill reference in path
    for i, path in enumerate(paths):
        fill = path.get('fill')
        if fill and fill.startswith('url(#'):
            gradient_id = fill[5:-1]  # get gradient id
            if gradient_id in gradients:
                # replace fill
                path.set('fill', gradients[gradient_id])
        elif fill is None:
            if fill_is_empty == 'previous':
                # If the current path does not fill, try to get the color from the previous valid path
                previous_colors = get_previous_fill_color(paths, i - 1)
                if previous_colors:
                    # Convert valid colors to RGB and calculate the average value
                    rgb_colors = [hex_to_rgb(color) for color in previous_colors]
                    average_hex_color = average_color(rgb_colors)
                    if average_hex_color:
                        path.set('fill', average_hex_color)
            elif fill_is_empty == 'skip':
                continue
            else:  # 'default': black
                path.set('fill', '#fffff')

    # delete all gradient
    for gradient in radial_gradients + linear_gradients:
        root.remove(gradient)

    # return etree.tostring(root, encoding=str)
    return etree.tostring(root, encoding="utf-8").decode()


"""Delete the <svg> tag and keep the other tags"""


def remove_svg_tag(svg_string):
    root = ET.fromstring(svg_string)

    if root.tag == '{http://www.w3.org/2000/svg}svg':
        result = ''

        # remove namespace. xmlns="http://www.w3.org/2000/svg"
        for elem in root.iter():
            elem.tag = elem.tag.split('}', 1)[1] if '}' in elem.tag else elem.tag

        # tostring
        for elem in root:
            result += ET.tostring(elem, encoding='unicode', method='xml')

        return result
    else:
        return svg_string


if __name__ == '__main__':
    svg_str = '''
    <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
        <circle cx="50" cy="50" r="40"/>
        <g fill="none" stroke="blue" stroke-width="2">
            <circle cx="50" cy="50" r="40"/>
            <g stroke="red">
                <rect x="10" y="10" width="30" height="30"/>
            </g>
        </g>
    </svg>
    '''
    svg_input = '''<svg xmlns="http://www.w3.org/2000/svg" width="128" height="128">
        <g transform="translate(10,10)">
            <rect x="106.66667" y="64" width="64" height="21.33333" rx="10.66666" ry="10.66666" fill="#aaaaa"/>
            <circle cx="405.33333" cy="64" r="21.33333" fill="#ffbe11"/>
            <ellipse width="150" height="150" x="10" y="10" rx="20" ry="20" fill="#ffffff" opacity="0.5"/>
            <path d="m53.72 42.6c-7.42-19.2-23.62-32.26-29.85-34.21-2.35-.74-5.3-.81-6.63 1.35-3.36 5.45-7.66 22.95 1.85 47.78z" fill="#ffc022"/>
        </g>
        <g fill="none" stroke="#9e9e9e" stroke-linecap="round" stroke-miterlimit="10" stroke-width="3">
            <path d="m122 78 s -5 -36 -16 2a 25 25 -30 1 1 50 -20" stroke="#ffc022"/>
            <path d="m122.42 78.49s-5.09-.36-16.05 1.97 a 25 25 -30 1 1 50 -20"/>
            <path d="m121.45 89.05 m1 s-4.83-1.71-14.78-2.25 s1 2"/>
        </g>
        <path d="m120.45 89.05 m1 s-4.83-1.71-14.78-2.25 s1 2"/>
        <path d="m120.45 89.05 m1 s-4.83-1.71-14.78-2.25 s1 2"/>
    </svg>'''

    result_svg = apply_g_attributes_to_children(svg_str)
    # result_svg = apply_g_attributes_to_children(svg_input)
    print(result_svg)
