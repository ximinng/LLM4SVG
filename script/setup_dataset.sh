#!/bin/bash

set -e

echo "Starting dataset setup..."

# Step 1: Ensure SVGX-rendering-data directory exists
RENDERING_DATA_DIR="./dataset/SVGX-rendering-data"
if [ ! -d "$RENDERING_DATA_DIR" ]; then
  echo "🔨 Creating directory '$RENDERING_DATA_DIR'..."
  mkdir -p "$RENDERING_DATA_DIR"
fi

# Step 2: Run the data processing Python script
echo "🚀 Running data processing script to export images..."
python SVGX-dataset/export_imgs.py --dataset_path dataset/SVGX-dataset/SVGX-Core-250k --output_file dataset/SVGX-rendering-data/ -a svg_to_png

echo "✅ Data processing completed."

# Step 3: Run the data processing Python script
echo "🚀 Running data processing script to export images..."
python SVGX-dataset/export_imgs.py --dataset_path dataset/SVGX-dataset/SVGX-Core-250k --output_file dataset/SVGX-rendering-data/ -a svg_to_png

echo "✅ Data processing completed."

# Step 4: Continue with dataset symbolic link setup
# Define dataset and rendering data paths
DATASET_DIR="./dataset/SVGX-dataset"
DATASET_LINK="./LLaMA-Factory/data/SVGX-dataset"
RENDERING_LINK="./LLaMA-Factory/data/SVGX-rendering-data"

# Ensure the parent directory for the links exists
if [ ! -d "./LLaMA-Factory/data/" ]; then
  echo "🔨 Creating directory './LLaMA-Factory/data/'..."
  mkdir -p ./LLaMA-Factory/data/
fi

# Function to create a symbolic link
create_symlink() {
  local target_dir=$1
  local link_path=$2
  local relative_target=$3

  if [ ! -d "$target_dir" ]; then
    echo "❌ Target directory '$target_dir' does not exist."
    exit 1
  fi

  if [ -L "$link_path" ]; then
    echo "🔗 Symbolic link already exists at '$link_path'. No action needed."
  elif [ -e "$link_path" ]; then
    echo "⚠️ A file or folder already exists at '$link_path'."
    echo "❌ Cannot create a symbolic link. Please remove or rename the existing item first."
    exit 1
  else
    echo "🔗 Creating symbolic link: '$link_path' -> '$relative_target'"
    ln -s "$relative_target" "$link_path"
    echo "✅ Symbolic link created."
  fi
}

# Create symbolic link for SVGX-dataset
create_symlink "$DATASET_DIR" "$DATASET_LINK" "../../dataset/SVGX-dataset"

# Create symbolic link for SVGX-rendering-data
create_symlink "$RENDERING_DATA_DIR" "$RENDERING_LINK" "../../dataset/SVGX-rendering-data"

echo "🎉 All setup completed successfully!"