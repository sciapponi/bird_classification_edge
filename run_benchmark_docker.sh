#!/bin/bash

# =============================================================================
# Bird Classification Benchmark Docker Runner
# =============================================================================
# Usage: ./run_benchmark_docker.sh [GPU_ID] [CONTAINER_NAME] [CONFIG_NAME]
#
# Arguments:
#   GPU_ID: GPU number to use (default: 0)
#   CONTAINER_NAME: Name for the docker container (default: benchmark_container)
#   CONFIG_NAME: Benchmark config file to use (default: benchmark)
#
# Examples:
#   ./run_benchmark_docker.sh                    # Use GPU 0, default config
#   ./run_benchmark_docker.sh 1                  # Use GPU 1, default config
#   ./run_benchmark_docker.sh 1 my_benchmark     # Use GPU 1, custom container name
#   ./run_benchmark_docker.sh 0 test quick_start # Use GPU 0, quick_start config
# =============================================================================

# Color output functions
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

print_success() {
    echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Default values
GPU_ID=${1:-0}
CONTAINER_NAME=${2:-"benchmark_container"}
CONFIG_NAME=${3:-"benchmark"}
IMAGE_NAME="bird_benchmark"

# Validation
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    print_error "GPU_ID must be a number. Got: $GPU_ID"
    exit 1
fi

# Check if NVIDIA GPU is available
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

# Check if specified GPU exists
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [ "$GPU_ID" -ge "$GPU_COUNT" ]; then
    print_error "GPU $GPU_ID not found. Available GPUs: 0-$((GPU_COUNT-1))"
    print_info "Available GPUs:"
    nvidia-smi --list-gpus
    exit 1
fi

print_info "=== Bird Classification Benchmark ==="
print_info "GPU ID: $GPU_ID"
print_info "Container: $CONTAINER_NAME"
print_info "Config: $CONFIG_NAME"
print_info "Image: $IMAGE_NAME"

# Check if Docker image exists
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
    print_warning "Docker image '$IMAGE_NAME' not found. Building..."
    
    # Get USER_ID and USER_GROUP_ID
    USER_ID=$(id -u)
    USER_GROUP_ID=$(id -g)
    
    print_info "Building Docker image with USER_ID=$USER_ID and USER_GROUP_ID=$USER_GROUP_ID"
    
    if ! docker build --no-cache \
        --build-arg USER_ID=$USER_ID \
        --build-arg USER_GROUP_ID=$USER_GROUP_ID \
        -t $IMAGE_NAME \
        -f Dockerfile.benchmark .; then
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    print_success "Docker image built successfully"
fi

# Stop and remove existing container if it exists
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    print_warning "Removing existing container: $CONTAINER_NAME"
    docker stop $CONTAINER_NAME 2>/dev/null
    docker rm $CONTAINER_NAME 2>/dev/null
fi

# Create output directory for results
mkdir -p benchmark_results

print_info "Starting benchmark container..."

# Run Docker container with GPU support
docker run \
    --name $CONTAINER_NAME \
    --gpus "device=$GPU_ID" \
    --rm \
    -it \
    --shm-size=32gb \
    -e "DOCKER_CONTAINER_NAME=$CONTAINER_NAME" \
    -e "CUDA_VISIBLE_DEVICES=$GPU_ID" \
    -e "PYTHONIOENCODING=utf-8" \
    -v $PWD:/workspace \
    -v $PWD/benchmark_results:/workspace/benchmark_results \
    -w /workspace \
    $IMAGE_NAME \
    bash -c "
        echo '🐦 Starting Bird Classification Benchmark 🐦'
        echo 'GPU Info:'
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader,nounits
        echo ''
        echo 'Starting benchmark with config: $CONFIG_NAME'
        echo 'CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES'
        echo ''
        
        # Run the benchmark
        cd /workspace && CUDA_VISIBLE_DEVICES=$GPU_ID python benchmark/run_benchmark.py --config-name=$CONFIG_NAME
        
        echo ''
        echo '✅ Benchmark completed!'
        echo 'Results saved in: benchmark_results/'
        
        # Keep container alive for inspection if needed
        echo ''
        echo 'Container will remain active for result inspection.'
        echo 'Press Ctrl+C to exit or use: docker stop $CONTAINER_NAME'
        echo ''
        bash
    "

print_success "Benchmark completed successfully!"
print_info "Results are available in: ./benchmark_results/" 