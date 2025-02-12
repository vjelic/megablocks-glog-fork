#!/bin/bash

# Get the location of torch using pip show
TORCH_DIR=$(pip show torch | grep "Location" | cut -d ' ' -f 2)

# Check if we found the location
if [ -z "$TORCH_DIR" ]; then
    echo "torch is not installed or pip is not available."
    exit 1
fi

# Change to the directory containing the torch package
cd "$TORCH_DIR"/torch/utils/hipify || { echo "Failed to change directory."; exit 1; }

# Path to the cuda_to_hip_mappings.py file
CUDA_TO_HIP_FILE="cuda_to_hip_mappings.py"

# Check if the file exists
if [ ! -f "$CUDA_TO_HIP_FILE" ]; then
    echo "$CUDA_TO_HIP_FILE does not exist in the directory."
    exit 1
fi

# Add the line to the file
sed -i '/("cub::CountingInputIterator", ("hipcub::CountingInputIterator", CONV_SPECIAL_FUNC, API_RUNTIME)),/a\
\        ("cub::DeviceHistogram", \("hipcub::DeviceHistogram", CONV_SPECIAL_FUNC, API_RUNTIME\)), ' "$CUDA_TO_HIP_FILE"
