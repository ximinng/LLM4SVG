# Dataset Download and Setup Instructions

This document outlines the steps to download and set up the required datasets.

## 1. Download Datasets

These commands download the necessary datasets using the HuggingFace CLI.

First, ensure the target directories exist:

```bash
# Create directory for the core dataset
mkdir -p dataset/SVGX-dataset/SVGX-Core-250k

# Create directory for the SFT dataset (if different, otherwise handled by the first command)
# Note: The script puts both under dataset/SVGX-dataset, with SVGX-Core-250k in a subdirectory.
# The mkdir for the parent is likely sufficient if the tool handles subdirs,
# but we include the specific target dir from the script for clarity.
mkdir -p dataset/SVGX-dataset
```

Now, download the datasets:

```shell
# Download SVGX-Core-250k
huggingface-cli download xingxm/SVGX-Core-250k --repo-type dataset --local-dir dataset/SVGX-dataset/SVGX-Core-250k --local-dir-use-symlinks False

# Download SVGX-SFT-1M
huggingface-cli download xingxm/SVGX-SFT-1M --repo-type dataset --local-dir dataset/SVGX-dataset --local-dir-use-symlinks False
```

## 2. Setup Dataset

These commands process the downloaded data and set up symbolic links.

First, create the directory for rendering data:

```shell
# Create directory for exported images
mkdir -p ./dataset/SVGX-rendering-data
```

Next, run the Python script to process the images (the script runs this command twice):

```shell
# Export images using the provided script (first run)
python SVGX-dataset/export_imgs.py --dataset_path dataset/SVGX-dataset/SVGX-Core-250k --output_file dataset/SVGX-rendering-data/ -a svg_to_png

# Export images using the provided script (second run)
python SVGX-dataset/export_imgs.py --dataset_path dataset/SVGX-dataset/SVGX-Core-250k --output_file dataset/SVGX-rendering-data/ -a svg_to_png
```

Finally, create the necessary symbolic links for LLaMA-Factory. Ensure the target directory for links exists:

```shell
# Create the data directory within LLaMA-Factory if it doesn't exist
mkdir -p ./LLaMA-Factory/data/
```

Create the symbolic links (ensure the target directories `./dataset/SVGX-dataset` and `./dataset/SVGX-rendering-data`
actually exist before running these):

```shell
# Link the main dataset directory
# (Note: adjust relative path if your LLaMA-Factory is not in the parent dir)
ln -s ../../dataset/SVGX-dataset ./LLaMA-Factory/data/SVGX-dataset

# Link the rendering data directory
# (Note: adjust relative path if your LLaMA-Factory is not in the parent dir)
ln -s ../../dataset/SVGX-rendering-data ./LLaMA-Factory/data/SVGX-rendering-data
```

## 3. Setup is complete.

- Directory Structure:

```
LLM4SVG/dataset/
├── SVGX-dataset/
│   └── SVGX_SFT_GEN_51k_encode.json          # Dataset part 1 
│   └── SVGX_SFT_vision_25k_encode.json       # Dataset part 2
│   └── ...                                   # More Files
├── SVGX-rendering-data/  
│   ├── 00004ed0-8ec5-4d1a-a9e8-898cd8f61235.png
│   ├── 0002bc4f-099b-4079-80c0-f6753e350e37.png
│   ├── 00042416-109a-4f53-975f-647d53b43a9f.png
│   └── ...  # more images
```