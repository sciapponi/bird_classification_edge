#!/bin/bash

# Test script to verify GPU availability in Docker container

echo "🔍 Testing GPU availability in Docker container..."

# Test with specific GPU
docker run --rm --gpus device=2 \
  -e CUDA_VISIBLE_DEVICES=2 \
  bird_classification_edge \
  python -c "
import torch
import sys

print('=== GPU Test Results ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('❌ CUDA not available - will use CPU')
    
# Test tensor creation
try:
    if torch.cuda.is_available():
        x = torch.randn(3, 3).cuda()
        print('✅ GPU tensor creation successful')
    else:
        print('⚠️  Using CPU tensors')
except Exception as e:
    print(f'❌ GPU tensor creation failed: {e}')
    
print('=== End GPU Test ===')
"

echo "✅ GPU test completed" 