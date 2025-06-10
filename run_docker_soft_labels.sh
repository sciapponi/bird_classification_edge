#!/bin/bash

# Script per eseguire l'estrazione di soft labels in Docker
# Uso: ./run_docker_soft_labels.sh CONTAINER_NAME [GPU_ID=0] [ADDITIONAL_ARGS...]

set -e

# Configuration
DOCKER_IMAGE_NAME="bird_classification_edge"
DEFAULT_CONTAINER_NAME="soft_labels_extraction"

# Directories to mount
HOST_PROJECT_DIR="$PWD"
HOST_BIRD_SOUND_DATASET_DIR="$PWD/bird_sound_dataset"
HOST_AUGMENTED_DATASET_DIR="$PWD/augmented_dataset"
HOST_ESC50_DIR="$PWD/esc-50"
HOST_SOFT_LABELS_OUTPUT_DIR="$PWD/soft_labels_complete"
HOST_LOGS_DIR="$PWD/logs"

# Container paths
CONTAINER_PROJECT_DIR="/app"
CONTAINER_BIRD_SOUND_DATASET_DIR="/app/bird_sound_dataset"
CONTAINER_AUGMENTED_DATASET_DIR="/app/augmented_dataset"
CONTAINER_ESC50_DIR="/app/esc-50"
CONTAINER_SOFT_LABELS_OUTPUT_DIR="/app/soft_labels_complete"
CONTAINER_LOGS_DIR="/app/logs"

# Parse arguments
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
GPU_ID=""
USE_GPU=true
ADDITIONAL_ARGS=()

# Process arguments
for arg in "$@"; do
    if [[ $arg == GPU_ID=* ]]; then
        GPU_ID="${arg#GPU_ID=}"
    elif [[ $arg == "MAC" ]]; then
        USE_GPU=false
    elif [[ -z $CONTAINER_NAME_SET ]]; then
        CONTAINER_NAME="$arg"
        CONTAINER_NAME_SET=true
    else
        ADDITIONAL_ARGS+=("$arg")
    fi
done

echo "🐳 DOCKER SOFT LABELS EXTRACTION"
echo "================================"
echo "Container name: $CONTAINER_NAME"
echo "Docker image: $DOCKER_IMAGE_NAME"

if [[ $USE_GPU == true ]]; then
    if [[ -n $GPU_ID ]]; then
        echo "GPU: $GPU_ID"
        GPU_FLAGS="--gpus device=$GPU_ID -e CUDA_VISIBLE_DEVICES=$GPU_ID"
    else
        echo "GPU: all available"
        GPU_FLAGS="--gpus all"
    fi
else
    echo "GPU: disabled (macOS mode)"
    GPU_FLAGS=""
fi

echo "Additional args: ${ADDITIONAL_ARGS[*]}"
echo ""

# Create output directories if they don't exist
mkdir -p "$HOST_SOFT_LABELS_OUTPUT_DIR"
mkdir -p "$HOST_LOGS_DIR"

# Verify required directories exist
echo "📂 Checking required directories..."
for dir in "$HOST_BIRD_SOUND_DATASET_DIR" "$HOST_AUGMENTED_DATASET_DIR" "$HOST_ESC50_DIR"; do
    if [[ ! -d "$dir" ]]; then
        echo "❌ ERROR: Required directory not found: $dir"
        exit 1
    else
        echo "✅ Found: $dir"
    fi
done

echo ""
echo "🚀 Starting soft labels extraction..."
echo "This will process ALL audio files in the dataset."
echo "Expected time: 2-3 hours for ~5000 files"
echo ""

# Build Docker command
DOCKER_CMD="docker run"
DOCKER_CMD="$DOCKER_CMD --name $CONTAINER_NAME"
DOCKER_CMD="$DOCKER_CMD --rm"
DOCKER_CMD="$DOCKER_CMD -it"
if [[ $USE_GPU == true ]]; then
    DOCKER_CMD="$DOCKER_CMD $GPU_FLAGS"
fi
DOCKER_CMD="$DOCKER_CMD --shm-size=8gb"
DOCKER_CMD="$DOCKER_CMD -e DOCKER_CONTAINER_NAME=$CONTAINER_NAME"

# Volume mounts
DOCKER_CMD="$DOCKER_CMD -v $HOST_PROJECT_DIR:$CONTAINER_PROJECT_DIR"
DOCKER_CMD="$DOCKER_CMD -v $HOST_BIRD_SOUND_DATASET_DIR:$CONTAINER_BIRD_SOUND_DATASET_DIR"
DOCKER_CMD="$DOCKER_CMD -v $HOST_AUGMENTED_DATASET_DIR:$CONTAINER_AUGMENTED_DATASET_DIR"
DOCKER_CMD="$DOCKER_CMD -v $HOST_ESC50_DIR:$CONTAINER_ESC50_DIR"
DOCKER_CMD="$DOCKER_CMD -v $HOST_SOFT_LABELS_OUTPUT_DIR:$CONTAINER_SOFT_LABELS_OUTPUT_DIR"
DOCKER_CMD="$DOCKER_CMD -v $HOST_LOGS_DIR:$CONTAINER_LOGS_DIR"

# Image and command
DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE_NAME"

# Default soft label extraction command
SOFT_LABELS_CMD="python extract_soft_labels.py"
SOFT_LABELS_CMD="$SOFT_LABELS_CMD --dataset_path bird_sound_dataset"
SOFT_LABELS_CMD="$SOFT_LABELS_CMD --output_path soft_labels_complete"
SOFT_LABELS_CMD="$SOFT_LABELS_CMD --confidence_threshold 0.05"

# Add any additional arguments
if [[ ${#ADDITIONAL_ARGS[@]} -gt 0 ]]; then
    SOFT_LABELS_CMD="$SOFT_LABELS_CMD ${ADDITIONAL_ARGS[*]}"
fi

# Final command
DOCKER_CMD="$DOCKER_CMD $SOFT_LABELS_CMD"

echo "📋 Docker command:"
echo "$DOCKER_CMD"
echo ""

# Execute
eval $DOCKER_CMD 