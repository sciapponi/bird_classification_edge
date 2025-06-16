#!/bin/bash

# Simple Bird Classification Benchmark Runner for Server
# Usage: ./run_benchmark_simple.sh [GPU_ID] [CONFIG_NAME]
# Examples:
#   ./run_benchmark_simple.sh 1                # Use GPU 1, default config
#   ./run_benchmark_simple.sh 0 quick_start    # Use GPU 0, quick_start config

GPU_ID=${1:-0}
CONFIG_NAME=${2:-"benchmark"}
IMAGE_NAME="bird_benchmark"
CONTAINER_NAME="benchmark_$(date +%Y%m%d_%H%M%S)"

echo "🐦 Bird Classification Benchmark 🐦"
echo "GPU: $GPU_ID | Config: $CONFIG_NAME | Container: $CONTAINER_NAME"

# Build image if it doesn't exist
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    echo "Building Docker image..."
    docker build --build-arg USER_ID=$(id -u) --build-arg USER_GROUP_ID=$(id -g) -t $IMAGE_NAME -f Dockerfile.benchmark .
fi

# Run benchmark
CUDA_VISIBLE_DEVICES=$GPU_ID docker run \
    --name $CONTAINER_NAME \
    --gpus "device=$GPU_ID" \
    --rm \
    -v $PWD:/workspace \
    -w /workspace \
    -e "CUDA_VISIBLE_DEVICES=$GPU_ID" \
    -e "PYTHONIOENCODING=utf-8" \
    $IMAGE_NAME \
    python benchmark/run_benchmark.py --config-name=$CONFIG_NAME

echo "✅ Benchmark completed! Check benchmark_results/ for outputs." 