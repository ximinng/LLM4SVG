# -*- coding: utf-8 -*-
# Author: ximing xing
# Copyright (c) 2025, XiMing Xing
# License: MIT License
# Description: Export the images from HuggingFace Dataset
# How to use:
# 1. PIL Image to PNG
# python export_imgs.py --dataset_path SVGX-Core-250k --output_file  SVGX-rendering-data
# 2. SVG code rendeing to PNG
# python export_imgs.py --dataset_path SVGX-Core-250k --output_file  SVGX-rendering-data -a svg_to_png
import argparse
import os
import uuid
from PIL import Image

import cairosvg
from datasets import load_from_disk


def save_images_from_dataset(dataset_path, output_dir="saved_images", image_field="image", uuid_field="uuid"):
    """
    Extracts images from a local Hugging Face dataset and saves them to disk.

    Parameters:
    - dataset_path (str): Path to the local Hugging Face dataset.
    - output_dir (str): Directory where images will be saved (default: "saved_images").
    - image_field (str): Name of the field containing PIL.Image objects (default: "image").
    - uuid_field (str): Name of the field used as the filename (default: "uuid").

    Returns:
    - None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset from the given path
    dataset = load_from_disk(dataset_path)

    # Iterate through the dataset and save images
    for idx, item in enumerate(dataset):
        img = item.get(image_field)  # Retrieve the image field
        img_uuid = item.get(uuid_field)  # Retrieve the UUID field

        # Ensure the image field contains a valid PIL.Image object
        if not isinstance(img, Image.Image):
            print(f"Warning: Image field at index {idx} is not a valid PIL.Image. Skipping.")
            continue

        # Ensure the UUID is a valid string
        if not isinstance(img_uuid, str):
            img_uuid = str(uuid.uuid4())

        # Construct the image file path
        img_path = os.path.join(output_dir, f"{img_uuid}.png")

        # Save the image to the specified directory
        img.save(img_path, format="PNG")

        print(f"Saved: {img_path}")

    print("All images have been successfully saved!")


def svg_to_png(dataset_path, output_dir="svg_rendered_images", svg_field="svg_code", uuid_field="uuid"):
    """
    Converts SVG code from a local Hugging Face dataset to PNG images and saves them to disk.

    Parameters:
    - dataset_path (str): Path to the local Hugging Face dataset.
    - output_dir (str): Directory where PNG images will be saved (default: "svg_rendered_images").
    - svg_field (str): Name of the field containing SVG code (default: "svg_code").
    - uuid_field (str): Name of the field used as the filename (default: "uuid").

    Returns:
    - None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset from the given path
    dataset = load_from_disk(dataset_path)

    # Iterate through the dataset and convert SVG to PNG
    for idx, item in enumerate(dataset):
        svg_code = item.get(svg_field)  # Retrieve the SVG code field
        img_uuid = item.get(uuid_field)  # Retrieve the UUID field

        # Skip if no SVG code is present
        if not svg_code or not isinstance(svg_code, str):
            print(f"Warning: No valid SVG code at index {idx}. Skipping.")
            continue

        # Ensure the UUID is a valid string
        if not isinstance(img_uuid, str):
            img_uuid = str(uuid.uuid4())  # Generate a new UUID if invalid

        # Construct the image file path
        img_path = os.path.join(output_dir, f"{img_uuid}.png")

        try:
            # Convert SVG to PNG using cairosvg
            png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))

            # Save the PNG data to file
            with open(img_path, 'wb') as f:
                f.write(png_data)

            print(f"Converted and saved: {img_path}")
        except Exception as e:
            print(f"Error converting SVG at index {idx}: {e}")
            continue

    print("All SVG to PNG conversions have been completed!")


def main():
    parser = argparse.ArgumentParser(description="Convert an SVG dataset into Alpaca format for LLM fine-tuning.")

    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the HuggingFace dataset directory")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the converted Alpaca JSON file")
    parser.add_argument("-a", "--action", type=str, default="export_pil_obj")

    args = parser.parse_args()

    if args.action == "export_pil_obj":
        save_images_from_dataset(args.dataset_path, args.output_file)
    elif args.action == "svg_to_png":
        svg_to_png(args.dataset_path, args.output_file)


if __name__ == '__main__':
    main()
