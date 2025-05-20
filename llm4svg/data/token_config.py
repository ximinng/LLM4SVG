# -*- coding: utf-8 -*-
# Author: ximing xing
# Copyright (c) 2025, XiMing Xing
# License: MIT License
# Description: SVG Semantic Tokens

SEMANTIC_SVG_TOKEN_MAPPER_DEFAULT = {
    # SVG Container Tags
    "[<|START_OF_SVG|>]": "Begin SVG document",
    "[<|END_OF_SVG|>]": "End SVG document",
    "[<|start_of_g|>]": "Begin SVG group",
    "[<|end_of_g|>]": "End SVG group",

    # SVG Shape Tags
    "[<|svg_path|>]": "SVG path element",
    "[<|svg_circle|>]": "SVG circle element",
    "[<|svg_rect|>]": "SVG rectangle element",
    "[<|svg_ellipse|>]": "SVG ellipse element",
    "[<|svg_polygon|>]": "SVG polygon element",
    "[<|svg_line|>]": "SVG line element",
    "[<|svg_polyline|>]": "SVG polyline element",
    "[<|svg_text|>]": "SVG text element",

    # SVG Gradient Tags
    "[<|svg_linearGradient|>]": "Linear gradient element",
    "[<|svg_radialGradient|>]": "Radial gradient element",
    "[<|svg_stop|>]": "Gradient stop element",

    # Path Commands
    "[<|moveto|>]": "Move to coordinate",
    "[<|lineto|>]": "Draw line to coordinate",
    "[<|horizontal_lineto|>]": "Draw horizontal line",
    "[<|vertical_lineto|>]": "Draw vertical line",
    "[<|curveto|>]": "Cubic Bézier curve",
    "[<|smooth_curveto|>]": "Smooth cubic Bézier curve",
    "[<|quadratic_bezier_curve|>]": "Quadratic Bézier curve",
    "[<|smooth_quadratic_bezier_curveto|>]": "Smooth quadratic Bézier curve",
    "[<|elliptical_Arc|>]": "Elliptical arc",
    "[<|close_the_path|>]": "Close path",

    # Attribute Tokens
    "[<|id|>]": "Element identifier",
    "[<|path_d|>]": "Path data",
    "[<|fill|>]": "Fill color",
    "[<|stroke-width|>]": "Stroke width",
    "[<|stroke-linecap|>]": "Stroke line cap",
    "[<|stroke|>]": "Stroke color",
    "[<|opacity|>]": "Opacity level",
    "[<|transform|>]": "Transform attributes",
    "[<|gradientTransform|>]": "Gradient transformation",
    "[<|offset|>]": "Gradient stop offset",
    "[<|width|>]": "Element width",
    "[<|height|>]": "Element height",
    "[<|cx|>]": "Circle center x-coordinate",
    "[<|cy|>]": "Circle center y-coordinate",
    "[<|rx|>]": "Ellipse x-radius",
    "[<|ry|>]": "Ellipse y-radius",
    "[<|r|>]": "Circle radius",
    "[<|points|>]": "Polygon or polyline points",
    "[<|x1|>]": "Start x-coordinate",
    "[<|y1|>]": "Start y-coordinate",
    "[<|x2|>]": "End x-coordinate",
    "[<|y2|>]": "End y-coordinate",
    "[<|x|>]": "X-coordinate",
    "[<|y|>]": "Y-coordinate",
    "[<|fr|>]": "Radial gradient focal radius",
    "[<|fx|>]": "Radial gradient focal x",
    "[<|fy|>]": "Radial gradient focal y",
    "[<|href|>]": "Reference link",
    "[<|rotate|>]": "Rotation angle",
    "[<|font-size|>]": "Font Size",
    "[<|font-style|>]": "Font Style",
    "[<|font-family|>]": "Font Family",
    "[<|text-content|>]": "Text Content"
}

SEMANTIC_SVG_TOKEN_MAPPER_ADVANCE = {
    # SVG Container Tags
    "[<|START_OF_SVG|>]": "Marks the beginning of an SVG document",
    "[<|END_OF_SVG|>]": "Marks the end of an SVG document",
    "[<|start_of_g|>]": "Begins a group of SVG elements (<g> tag)",
    "[<|end_of_g|>]": "Ends a group of SVG elements",

    # SVG Shape Tags
    "[<|svg_path|>]": "Defines a path consisting of lines, curves, and arcs",
    "[<|svg_circle|>]": "Represents a circle with a center (cx, cy) and a radius (r)",
    "[<|svg_rect|>]": "Defines a rectangle with width, height, and optional rounded corners",
    "[<|svg_ellipse|>]": "Represents an ellipse with radii (rx, ry) and a center (cx, cy)",
    "[<|svg_polygon|>]": "Defines a closed shape with multiple points",
    "[<|svg_line|>]": "Draws a straight line between two points (x1, y1) to (x2, y2)",
    "[<|svg_polyline|>]": "Defines a series of connected lines",
    "[<|svg_text|>]": "Represents text within an SVG",

    # SVG Gradient Tags
    "[<|svg_linearGradient|>]": "Defines a linear gradient for filling shapes (color transition along a line)",
    "[<|svg_radialGradient|>]": "Defines a radial gradient for filling shapes (color transition outward from a center)",
    "[<|svg_stop|>]": "Defines a color stop within a gradient",

    # Path Commands
    "[<|moveto|>]": "Moves the drawing cursor to a new position (e.g., 'M 100 200')",
    "[<|lineto|>]": "Draws a line from the current position to a new point (e.g., 'L 150 250')",
    "[<|horizontal_lineto|>]": "Draws a horizontal line to a new x-coordinate (e.g., 'H 300')",
    "[<|vertical_lineto|>]": "Draws a vertical line to a new y-coordinate (e.g., 'V 400')",
    "[<|curveto|>]": "Draws a cubic Bézier curve (e.g., 'C x1 y1, x2 y2, x y')",
    "[<|smooth_curveto|>]": "Draws a smooth cubic Bézier curve without specifying the first control point",
    "[<|quadratic_bezier_curve|>]": "Draws a quadratic Bézier curve (e.g., 'Q x1 y1, x y')",
    "[<|smooth_quadratic_bezier_curveto|>]": "Draws a smooth quadratic Bézier curve without specifying the first control point",
    "[<|elliptical_Arc|>]": "Draws an elliptical arc with radii, rotation, and direction",
    "[<|close_the_path|>]": "Closes the current path by drawing a line back to the start (e.g., 'Z')",

    # Attribute Tokens
    "[<|id|>]": "Specifies a unique identifier for an SVG element",
    "[<|path_d|>]": "Defines the path data for an SVG <path> element",
    "[<|fill|>]": "Sets the fill color of an element (e.g., 'red', 'none', '#ff0000')",
    "[<|stroke-width|>]": "Defines the width of the stroke (outline) of a shape",
    "[<|stroke-linecap|>]": "Defines the shape of the end of a stroke (e.g., 'butt', 'round', 'square')",
    "[<|stroke|>]": "Sets the stroke (outline) color of a shape",
    "[<|opacity|>]": "Defines the transparency level (0.0 - fully transparent, 1.0 - fully opaque)",
    "[<|transform|>]": "Applies transformations such as rotate, scale, or translate",
    "[<|gradientTransform|>]": "Applies transformations to gradients",
    "[<|offset|>]": "Defines the position of a color stop in a gradient (0% to 100%)",
    "[<|width|>]": "Specifies the width of an element",
    "[<|height|>]": "Specifies the height of an element",
    "[<|cx|>]": "Defines the x-coordinate of the center of a circle or ellipse",
    "[<|cy|>]": "Defines the y-coordinate of the center of a circle or ellipse",
    "[<|rx|>]": "Specifies the x-radius of an ellipse",
    "[<|ry|>]": "Specifies the y-radius of an ellipse",
    "[<|r|>]": "Defines the radius of a circle",
    "[<|points|>]": "Defines the points of a polygon or polyline",
    "[<|x1|>]": "Specifies the starting x-coordinate of a line",
    "[<|y1|>]": "Specifies the starting y-coordinate of a line",
    "[<|x2|>]": "Specifies the ending x-coordinate of a line",
    "[<|y2|>]": "Specifies the ending y-coordinate of a line",
    "[<|x|>]": "Defines the x-coordinate of an element",
    "[<|y|>]": "Defines the y-coordinate of an element",
    "[<|fr|>]": "Defines the focal radius in a radial gradient",
    "[<|fx|>]": "Defines the x-coordinate of the focal point in a radial gradient",
    "[<|fy|>]": "Defines the y-coordinate of the focal point in a radial gradient",
    "[<|href|>]": "References an external resource (e.g., a linked gradient or pattern)",
    "[<|rotate|>]": "Defines the rotation angle of an element",
    "[<|font-size|>]": "Specifies the font size for text elements",
    "[<|font-style|>]": "Set the font style of the text",
    "[<|font-family|>]": "Set the font family of the text",
    "[<|text-content|>]": "Set the text content"
}