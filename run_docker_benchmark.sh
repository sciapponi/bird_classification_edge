#!/bin/bash

# --- Configuration - MODIFY THESE VARIABLES IF NECESSARY ---
IMAGE_NAME="bird_classification_edge"
CONTAINER_NAME_DEFAULT="bird_benchmark_container" # Default container name

# Paths relative to script location (assuming datasets are in project root)
HOST_BIRD_SOUND_DATASET_DIR="$PWD/bird_sound_dataset"
HOST_AUGMENTED_DATASET_DIR="$PWD/augmented_dataset"
HOST_BENCHMARK_DIR="$PWD/benchmark"
HOST_LOGS_DIR="$PWD/logs"
HOST_MODELS_DIR="$PWD" # For best_distillation_model.pt in root

# Paths inside container (generally no need to modify if Dockerfile uses /app)
CONTAINER_APP_DIR="/app"
CONTAINER_BIRD_SOUND_DATASET_DIR="$CONTAINER_APP_DIR/bird_sound_dataset"
CONTAINER_AUGMENTED_DATASET_DIR="$CONTAINER_APP_DIR/augmented_dataset"
CONTAINER_BENCHMARK_DIR="$CONTAINER_APP_DIR/benchmark"
CONTAINER_LOGS_DIR="$CONTAINER_APP_DIR/logs"

# --- End Configuration ---

# Argument handling
CONTAINER_NAME="$CONTAINER_NAME_DEFAULT"
GPU_ID=""
EXTRA_HYDRA_PARAMS=""
USE_GPU=true # Changed from RUN_ON_MAC to USE_GPU for consistency
BENCHMARK_CONFIG="quick_start" # Default benchmark config

# Argument parsing for container name, GPU, and Hydra parameters
# Example: ./run_docker_benchmark.sh my_container GPU_ID=0 debug.files_limit=10
# For Mac: ./run_docker_benchmark.sh my_container MAC debug.files_limit=10
# Custom config: ./run_docker_benchmark.sh my_container CONFIG=benchmark debug.files_limit=50
if [ "$#" -ge 1 ]; then
    CONTAINER_NAME=$1
    shift # Remove first argument (container name)
    while (( "$#" )); do
        if [[ "$1" == "GPU_ID="* ]]; then
            GPU_ID="${1#GPU_ID=}"
        elif [[ "$1" == "CONFIG="* ]]; then
            BENCHMARK_CONFIG="${1#CONFIG=}"
        elif [[ "$1" == "MAC" ]]; then
            USE_GPU=false
        else
            EXTRA_HYDRA_PARAMS="$EXTRA_HYDRA_PARAMS $1"
        fi
        shift
    done
fi

echo "--- Starting Docker Benchmark Container ---"
echo "Container Name: $CONTAINER_NAME"
echo "Benchmark Config: $BENCHMARK_CONFIG"

# GPU configuration (copying working distillation script pattern exactly)
if [[ $USE_GPU == true ]]; then
    if [[ -n $GPU_ID ]]; then
        echo "GPU: $GPU_ID"
        GPU_FLAGS="--gpus device=$GPU_ID"
    else
        echo "GPU: all available"
        GPU_FLAGS="--gpus all"
    fi
else
    echo "GPU: disabled (Mac mode)"
    GPU_FLAGS=""
fi

echo "Additional Hydra Parameters: ${EXTRA_HYDRA_PARAMS:-None}"
echo "----------------------------------------"

# Build volume mounts
VOLUME_MOUNTS=""

# Check directory existence before adding mounts
if [ -d "$HOST_BIRD_SOUND_DATASET_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_BIRD_SOUND_DATASET_DIR:$CONTAINER_BIRD_SOUND_DATASET_DIR"
else
    echo "Warning: bird_sound_dataset directory not found at $HOST_BIRD_SOUND_DATASET_DIR"
fi

if [ -d "$HOST_AUGMENTED_DATASET_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_AUGMENTED_DATASET_DIR:$CONTAINER_AUGMENTED_DATASET_DIR"
else
    echo "Warning: augmented_dataset directory not found at $HOST_AUGMENTED_DATASET_DIR"
fi

if [ -d "$HOST_BENCHMARK_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_BENCHMARK_DIR:$CONTAINER_BENCHMARK_DIR"
else
    echo "CRITICAL ERROR: benchmark directory not found at $HOST_BENCHMARK_DIR. Cannot continue."
    exit 1
fi

# Create logs directory on host if it doesn't exist
if [ ! -d "$HOST_LOGS_DIR" ]; then
    echo "Creating logs directory at $HOST_LOGS_DIR"
    mkdir -p "$HOST_LOGS_DIR"
fi
VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_LOGS_DIR:$CONTAINER_LOGS_DIR"

# Mount model file (best_distillation_model.pt)
VOLUME_MOUNTS="$VOLUME_MOUNTS -v $HOST_MODELS_DIR/best_distillation_model.pt:$CONTAINER_APP_DIR/best_distillation_model.pt"

# Check if model file exists
if [ ! -f "$HOST_MODELS_DIR/best_distillation_model.pt" ]; then
    echo "CRITICAL ERROR: Model file not found at $HOST_MODELS_DIR/best_distillation_model.pt. Cannot continue."
    exit 1
fi

echo "Volume mounts configured:"
echo "$VOLUME_MOUNTS"
echo "----------------------------------------"

# Docker run command
echo "Starting benchmark with config: $BENCHMARK_CONFIG"
echo "Full command: docker run $GPU_FLAGS --name $CONTAINER_NAME --rm -it --shm-size=16gb $VOLUME_MOUNTS $IMAGE_NAME python benchmark/run_benchmark.py --config-name=$BENCHMARK_CONFIG $EXTRA_HYDRA_PARAMS"

# shellcheck disable=SC2086
docker run $GPU_FLAGS --name "$CONTAINER_NAME" --rm -it --shm-size=16gb \
    $VOLUME_MOUNTS \
    "$IMAGE_NAME" \
    python benchmark/run_benchmark.py --config-name="$BENCHMARK_CONFIG" $EXTRA_HYDRA_PARAMS

echo "--- Docker Benchmark Container execution completed ---" 