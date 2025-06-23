#!/bin/bash

# Script to run ALIGNED benchmark with Docker using dedicated GPU
# This ensures fair comparison between BirdNET and student model using identical preprocessing
# Supports all new configurations: statistical analysis, multiple runs, birds-only, adaptive thresholds
#
# Usage: ./run_docker_benchmark.sh <container_name> <gpu_id> [config_name] [additional_hydra_overrides...]
#
# Configuration Examples:
#   ./run_docker_benchmark.sh my_benchmark_gpu0 0 benchmark
#   ./run_docker_benchmark.sh my_stats_gpu0 0 statistical_analysis
#   ./run_docker_benchmark.sh my_multi_gpu1 1 multiple_runs_benchmark
#   ./run_docker_benchmark.sh my_birds_gpu2 2 birds_only_benchmark
#   ./run_docker_benchmark.sh my_adaptive_gpu0 0 adaptive_threshold_benchmark
#   ./run_docker_benchmark.sh my_test_gpu1 1 test_benchmark

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <container_name> <gpu_id> [config_name] [additional_hydra_overrides...]"
    echo ""
    echo "Available Configurations:"
    echo "  benchmark                    - Standard benchmark"
    echo "  statistical_analysis         - Sample size calculation and power analysis"
    echo "  multiple_runs_benchmark      - Multiple runs with confidence intervals"
    echo "  birds_only_benchmark         - 8-class comparison (no no_birds)"
    echo "  adaptive_threshold_benchmark - Optimized thresholds to reduce false positives"
    echo "  test_benchmark               - Quick test mode"
    echo ""
    echo "Examples:"
    echo "  $0 my_benchmark_gpu0 0 benchmark"
    echo "  $0 my_stats_gpu0 0 statistical_analysis"
    echo "  $0 my_multi_gpu1 1 multiple_runs_benchmark"
    echo "  $0 my_birds_gpu2 2 birds_only_benchmark debug.files_limit=100"
    exit 1
fi

CONTAINER_NAME=$1
GPU_ID=$2
CONFIG_NAME=${3:-benchmark}  # Default to 'benchmark' if not specified
shift 2
if [ $# -gt 0 ] && [[ "$1" != *"="* ]]; then
    # Third argument is config name, not a hydra override
    shift 1
fi

# Get current user ID and group ID for proper file permissions
USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Prepare hydra overrides as space-separated string
HYDRA_OVERRIDES=""
if [ $# -gt 0 ]; then
    HYDRA_OVERRIDES="$@"
fi

echo "🎯 Starting ALIGNED benchmark container: $CONTAINER_NAME"
echo "🔧 Fair comparison mode: BirdNET and Student use identical preprocessing"
echo "⚙️  Configuration: $CONFIG_NAME"
echo "🖥️  Using GPU: $GPU_ID"
echo "👤 User ID: $USER_ID, Group ID: $GROUP_ID"
if [ -n "$HYDRA_OVERRIDES" ]; then
    echo "🔧 Hydra overrides: $HYDRA_OVERRIDES"
fi

# Build image with user arguments if it doesn't exist
if ! docker image inspect bird_classification_edge_benchmark >/dev/null 2>&1; then
    echo "Building benchmark Docker image with user ID $USER_ID..."
    docker build -f Dockerfile.benchmark \
        --build-arg USER_ID="$USER_ID" \
        --build-arg USER_GROUP_ID="$GROUP_ID" \
        -t bird_classification_edge_benchmark .
fi

# Check if GPU is available and set appropriate flags
if nvidia-smi >/dev/null 2>&1; then
    GPU_FLAGS="--gpus device=$GPU_ID"
    echo "🖥️  GPU detected, using GPU $GPU_ID"
else
    GPU_FLAGS=""
    echo "🖥️  No GPU detected, running on CPU"
fi

# Run the container with benchmark
docker run -it --rm \
    $GPU_FLAGS \
    --name "$CONTAINER_NAME" \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    -e PYTHONIOENCODING=utf-8 \
    -v "$(pwd):/workspace" \
    -w /workspace \
    bird_classification_edge_benchmark \
    bash -c "cd benchmark && python run_benchmark.py --config-name=$CONFIG_NAME $HYDRA_OVERRIDES"

# Configuration-specific completion messages
if [ "$CONFIG_NAME" = "statistical_analysis" ]; then
    echo "🎉 Statistical Analysis completed!"
    echo "📊 Sample size recommendations and power analysis available"
    echo "📁 Output saved to: benchmark/benchmark_results/statistical_analysis_YYYY-MM-DD_HH-MM-SS/comparison/"
    echo "📄 Files created:"
    echo "   • statistical_analysis.json - Detailed analysis data"
    echo "   • Check Docker logs above for summary results"
elif [ "$CONFIG_NAME" = "multiple_runs_benchmark" ]; then
    echo "🎉 Multiple Runs Benchmark completed!"
    echo "📊 Confidence intervals and statistical validation performed"
    echo "📈 Check benchmark/benchmark_results/multiple_runs_*/ for detailed analysis"
    echo "   Each run has its own timestamped directory + aggregated results"
else
    echo "🎉 ALIGNED benchmark ($CONFIG_NAME) completed!"
echo "✅ Fair comparison achieved - both models used identical preprocessing"
    echo "📁 Results saved in timestamped directory: benchmark/benchmark_results/YYYY-MM-DD_HH-MM-SS/"
    echo "📊 Check the comparison/ subdirectory for detailed analysis"
fi 