#!/bin/bash

# Script per eseguire il training con knowledge distillation in Docker
# Uso: ./run_docker_distillation.sh CONTAINER_NAME [GPU_ID=0] [HYDRA_OVERRIDES...]

set -e

# Configuration
DOCKER_IMAGE_NAME="bird_classification_edge"
DEFAULT_CONTAINER_NAME="distillation_training"

# Directories to mount
HOST_PROJECT_DIR="$PWD"
HOST_BIRD_SOUND_DATASET_DIR="$PWD/bird_sound_dataset"
HOST_AUGMENTED_DATASET_DIR="$PWD/augmented_dataset"
HOST_ESC50_DIR="$PWD/esc-50"
HOST_SOFT_LABELS_DIR="$PWD/test_soft_labels"
HOST_LOGS_DIR="$PWD/logs"

# Container paths
CONTAINER_PROJECT_DIR="/app"
CONTAINER_BIRD_SOUND_DATASET_DIR="/app/bird_sound_dataset"
CONTAINER_AUGMENTED_DATASET_DIR="/app/augmented_dataset"
CONTAINER_ESC50_DIR="/app/esc-50"
CONTAINER_SOFT_LABELS_DIR="/app/test_soft_labels"
CONTAINER_LOGS_DIR="/app/logs"

# Parse arguments
CONTAINER_NAME="$DEFAULT_CONTAINER_NAME"
GPU_ID=""
USE_GPU=true
HYDRA_OVERRIDES=()

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
        HYDRA_OVERRIDES+=("$arg")
    fi
done

echo "🐳 DOCKER KNOWLEDGE DISTILLATION TRAINING"
echo "========================================="
echo "Container name: $CONTAINER_NAME"
echo "Docker image: $DOCKER_IMAGE_NAME"

if [[ $USE_GPU == true ]]; then
    if [[ -n $GPU_ID ]]; then
        echo "GPU: $GPU_ID"
        GPU_FLAGS="--gpus device=$GPU_ID"
    else
        echo "GPU: all available"
        GPU_FLAGS="--gpus all"
    fi
else
    echo "GPU: disabled (macOS mode)"
    GPU_FLAGS=""
fi

echo "Hydra overrides: ${HYDRA_OVERRIDES[*]}"
echo ""

# Create output directories if they don't exist
mkdir -p "$HOST_LOGS_DIR"

# Verify required directories exist
echo "📂 Checking required directories..."
REQUIRED_DIRS=(
    "$HOST_BIRD_SOUND_DATASET_DIR"
    "$HOST_AUGMENTED_DATASET_DIR" 
    "$HOST_ESC50_DIR"
    "$HOST_SOFT_LABELS_DIR"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [[ ! -d "$dir" ]]; then
        echo "❌ ERROR: Required directory not found: $dir"
        if [[ "$dir" == "$HOST_SOFT_LABELS_DIR" ]]; then
            echo "   💡 Hint: Run soft labels extraction first with:"
            echo "   ./run_docker_soft_labels.sh your_extraction_container_name GPU_ID=0"
        fi
        exit 1
    else
        echo "✅ Found: $dir"
    fi
done

# Check if soft labels files exist
if [[ ! -f "$HOST_SOFT_LABELS_DIR/soft_labels.json" ]]; then
    echo "❌ ERROR: Soft labels file not found: $HOST_SOFT_LABELS_DIR/soft_labels.json"
    echo "   💡 Run soft labels extraction first!"
    exit 1
else
    echo "✅ Found soft labels file"
fi

echo ""
echo "🚀 Starting knowledge distillation training..."
echo "Expected time: varies based on epochs and dataset size"
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
DOCKER_CMD="$DOCKER_CMD -v $HOST_SOFT_LABELS_DIR:$CONTAINER_SOFT_LABELS_DIR"
DOCKER_CMD="$DOCKER_CMD -v $HOST_LOGS_DIR:$CONTAINER_LOGS_DIR"

# Image and command
DOCKER_CMD="$DOCKER_CMD $DOCKER_IMAGE_NAME"

# Default distillation command
DISTILLATION_CMD="python train_distillation.py"
# The path inside the container must match the mounted volume
DISTILLATION_CMD="$DISTILLATION_CMD dataset.soft_labels_path=test_soft_labels"

# Add any Hydra overrides
if [[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]]; then
    DISTILLATION_CMD="$DISTILLATION_CMD ${HYDRA_OVERRIDES[*]}"
fi

# Final command
DOCKER_CMD="$DOCKER_CMD $DISTILLATION_CMD"

echo "📋 Docker command:"
echo "$DOCKER_CMD"
echo ""

# Execute
eval $DOCKER_CMD 