#!/bin/bash

set -e

echo "Starting dataset download using HuggingFace CLI..."

# List of datasets to download (repo name ➔ target local directory)
declare -A datasets=(
  ["xingxm/SVGX-Core-250k"]="dataset/SVGX-dataset/SVGX-Core-250k"
  ["xingxm/SVGX-SFT-1M"]="dataset/SVGX-dataset"
)

# Loop through all datasets
for repo in "${!datasets[@]}"; do
  local_dir="${datasets[$repo]}"

  echo "📦 Downloading dataset '$repo' into '$local_dir'..."

  # Check if local directory exists
  if [ ! -d "$local_dir" ]; then
    mkdir -p "$local_dir"
  fi

  # Download using huggingface-cli
  huggingface-cli download "$repo" --repo-type dataset --local-dir "$local_dir" --local-dir-use-symlinks False

  echo "✅ Successfully downloaded '$repo'."
done

echo "🎉 All datasets have been downloaded successfully!"