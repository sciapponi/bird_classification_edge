#!/bin/bash

# Script to run benchmark with Docker using dedicated GPU
# Usage: ./run_docker_benchmark.sh <container_name> <gpu_id> [additional_hydra_overrides...]
#
# Examples:
#   ./run_docker_benchmark.sh my_benchmark_gpu0 0
#   ./run_docker_benchmark.sh my_benchmark_gpu1 1 benchmark.paths.student_model=my_model.pt
#   ./run_docker_benchmark.sh my_benchmark_gpu2 2 benchmark.files_limit=50

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <container_name> <gpu_id> [additional_hydra_overrides...]"
    echo ""
    echo "Examples:"
    echo "  $0 my_benchmark_gpu0 0"
    echo "  $0 my_benchmark_gpu1 1 benchmark.paths.student_model=my_model.pt"
    echo "  $0 my_benchmark_gpu2 2 benchmark.files_limit=50"
    exit 1
fi

CONTAINER_NAME=$1
GPU_ID=$2
shift 2  # Remove first two arguments, remaining ones are hydra overrides

# Get current user ID and group ID for proper file permissions
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Prepare hydra overrides as space-separated string
HYDRA_OVERRIDES=""
if [ $# -gt 0 ]; then
    HYDRA_OVERRIDES="$@"
fi

echo "Starting benchmark container: $CONTAINER_NAME"
echo "Using GPU: $GPU_ID"
echo "User ID: $USER_ID, Group ID: $GROUP_ID"
if [ -n "$HYDRA_OVERRIDES" ]; then
    echo "Hydra overrides: $HYDRA_OVERRIDES"
fi

# Build image with user arguments if it doesn't exist
if ! docker image inspect bird_classification_edge_benchmark >/dev/null 2>&1; then
    echo "Building benchmark Docker image with user ID $USER_ID..."
    docker build -f Dockerfile.benchmark \
        --build-arg USER_ID="$USER_ID" \
        --build-arg USER_GROUP_ID="$GROUP_ID" \
        -t bird_classification_edge_benchmark .
fi

# Run the container with benchmark
docker run -it --rm \
    --gpus "device=$GPU_ID" \
    --name "$CONTAINER_NAME" \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    -e PYTHONIOENCODING=utf-8 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    bird_classification_edge_benchmark \
    bash -c "cd benchmark && python run_benchmark.py --config-name=quick_start $HYDRA_OVERRIDES"

echo "Benchmark completed. Results saved in benchmark/benchmark_results/" 