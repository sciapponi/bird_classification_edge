# Bird Sound Classification on Edge Devices

This repository contains a comprehensive bird sound classification system designed to run on edge devices. It includes functionality for dataset preparation, augmentation, training, knowledge distillation, and benchmarking.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Standard Training Workflow](#standard-training-workflow)
- [Advanced Workflow: Knowledge Distillation](#advanced-workflow-knowledge-distillation)
- [Advanced Loss Functions: Focal Loss](#advanced-loss-functions-focal-loss)
- [Model Benchmarking System](#model-benchmarking-system)
- [Docker Execution](#docker-execution)
- [Configuration Reference](#configuration-reference)

## Quick Start

**Get started in 3 steps:**

1. **Setup Environment:**
   ```bash
   git clone <repository_url>
   cd bird_classification_edge
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare Dataset:**
   ```bash
   # Place your bird recordings in bird_sound_dataset/species_name/
   # Generate "no birds" samples
   python generate_no_birds_samples.py
   ```

3. **Train Model:**
   ```bash
   # Standard training
   python train.py
   
   # Or with knowledge distillation
   python extract_soft_labels.py
   python train_distillation.py
   ```

4. **Benchmark Your Model:**
   ```bash
   # Quick test (10 files)
   ./run_docker_benchmark.sh my_test 1 debug.files_limit=10
   
   # Full benchmark
   ./run_docker_benchmark.sh my_benchmark 1
   ```

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Docker (for containerized execution)

### 1. Clone and Setup Environment
    ```bash
    git clone <repository_url>
    cd bird_classification_edge
    python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

### 2. Dataset Preparation
- **Bird Recordings**: Place in `bird_sound_dataset/`, organized by species folders
- **Environmental Sounds**: ESC-50 dataset downloads automatically on first run
- **No-Birds Samples**: Generate using `python generate_no_birds_samples.py`

### 3. Verify Installation
```bash
# Test basic functionality
python train.py training.epochs=1 debug.files_limit=10
```

## Project Structure

```
bird_classification_edge/
├── Audio Data
│   ├── bird_sound_dataset/          # Your bird recordings (by species)
│   ├── augmented_dataset/           # Generated "no birds" samples
│   └── esc-50/                      # ESC-50 environmental sounds
│
├── Core Training
│   ├── train.py                     # Main training script
│   ├── models.py                    # Neural network architectures
│   ├── modules.py                   # Model components (GRU, attention, etc.)
│   └── config/                      # Training configurations
│
├── Datasets
│   ├── datasets/
│   │   ├── bird_dataset.py          # Bird sound dataset loader
│   │   ├── esc50_dataset.py         # ESC-50 environmental sounds
│   │   ├── empty_segment_dataset.py # Silent segment extraction
│   │   ├── dataset_factory.py       # Combined dataset creation
│   │   └── audio_utils.py           # Audio processing utilities
│   └── generate_no_birds_samples.py # Offline "no birds" generation
│
├── Knowledge Distillation
│   ├── distillation/
│   │   ├── scripts/                 # Distillation execution scripts
│   │   ├── datasets/                # Soft label dataset loaders
│   │   ├── losses/                  # Advanced loss functions (focal, distillation)
│   │   └── configs/                 # 8 comprehensive training configurations
│   ├── extract_soft_labels.py       # Extract BirdNET soft labels
│   └── train_distillation.py        # Train with knowledge distillation & focal loss
│
├── Benchmarking
│   ├── benchmark/
│   │   ├── run_benchmark.py         # Main benchmark orchestrator
│   │   ├── predict_student.py       # Student model predictions
│   │   ├── predict_birdnet.py       # BirdNET reference predictions
│   │   ├── compare_predictions.py   # Performance comparison & metrics
│   │   └── config/                  # Benchmark configurations
│   └── run_docker_benchmark.sh      # Docker benchmark execution
│
├── Docker & Deployment
│   ├── Dockerfile                   # Main training container
│   ├── Dockerfile.benchmark         # Benchmark container
│   ├── run_docker_*.sh              # Docker execution scripts
│   └── docker-compose.yml           # Multi-container orchestration
│
└── Documentation
    ├── README.md                    # This file
    ├── benchmark/README.md          # Benchmark system guide
    └── distillation/README.md       # Knowledge distillation guide
```

## Standard Training Workflow

Train a bird classification model using standard supervised learning.

### 1. Configure Training
Edit `config/bird_classification.yaml`:
```yaml
training:
  epochs: 100
  batch_size: 32
  
dataset:
  allowed_bird_classes: ["Bubo_bubo", "Apus_apus", "Certhia_familiaris"]
  load_pregenerated_no_birds: true  # Use pre-generated samples
  
optimizer:
  lr: 0.001
```

### 2. Handle "No Birds" Samples

**Option A: Offline Generation (Recommended)**
```bash
# Generate fixed set of "no birds" samples
python generate_no_birds_samples.py \
  --num_samples 500 \
  --esc50_ratio 0.5 \
  --energy_threshold_factor 1.5

# Configure to use pre-generated samples
# Set load_pregenerated_no_birds: true in config
```

**Option B: Online Generation**
    ```bash
# Set load_pregenerated_no_birds: false in config
# Samples generated during training setup
    ```

### 3. Train the Model
```bash
# Basic training
python train.py

# With parameter overrides
python train.py training.epochs=50 optimizer.lr=0.001 dataset.num_workers=8

# GPU selection
CUDA_VISIBLE_DEVICES=1 python train.py
```

### 4. Monitor Training
- Logs: `logs/` directory
- Models: Saved as `*.pt` files
- Metrics: Displayed during training

## Advanced Workflow: Knowledge Distillation

Improve model performance using BirdNET as a teacher model.

### How Knowledge Distillation Works
The student model learns from two sources:
1. **Hard Labels**: Ground truth species labels
2. **Soft Labels**: Teacher model's probability distributions

This is controlled by: `L_total = (1-α) × L_hard + α × L_soft`

### Advanced Loss Functions for Class Imbalance

The system now supports multiple loss function types to handle class imbalance:

- **Standard Distillation**: Cross-entropy + knowledge distillation
- **Focal Loss**: Down-weights easy examples, emphasizes hard ones
- **Focal Distillation**: Combines focal loss with knowledge distillation
- **Adaptive Focal**: Automatically adjusts gamma based on class distribution

**Configuration Options:**
```yaml
loss:
  type: "focal_distillation"  # Options: "distillation", "focal", "focal_distillation"
  gamma: 2.0                  # Focusing parameter (higher = more focus on hard examples)
  class_weights: "auto"       # Options: null, "auto", or manual list
  alpha_scaling: 1.0          # Weight scaling factor

distillation:
  alpha: 0.3                  # Balance between hard and soft loss
  temperature: 4.0            # Distillation temperature
```

### Workflow Steps

#### Step 1: Extract Soft Labels from BirdNET
```bash
# Extract for all species in distillation/species.txt
python extract_soft_labels.py --output_path soft_labels_complete

# Extract for custom species subset
python extract_soft_labels.py \
  --species_list distillation/species_4.txt \
  --output_path soft_labels_4_classes
```

#### Step 2: Train with Distillation
```bash
# Standard knowledge distillation
python train_distillation.py --config-name=distillation_config

# Focal loss + distillation (for imbalanced data)
python train_distillation.py --config-name=focal_loss_config

# Pure focal loss (no teacher model needed)
python train_distillation.py --config-name=pure_focal_config

# Adaptive focal for severe imbalance
python train_distillation.py --config-name=adaptive_focal_config

# With custom parameters
python train_distillation.py --config-name=focal_loss_config \
  training.alpha=0.7 \
  training.temperature=4.0 \
  training.epochs=50
```

### Custom Species Training
To train on a subset of species:

1. **Create Species List:**
   ```bash
   # Create distillation/species_custom.txt
   echo "Poecile montanus" > distillation/species_custom.txt
   echo "Certhia familiaris" >> distillation/species_custom.txt
   echo "Apus apus" >> distillation/species_custom.txt
   echo "Bubo bubo" >> distillation/species_custom.txt
    ```

2. **Extract Matching Soft Labels:**
    ```bash
    python extract_soft_labels.py \
     --species_list distillation/species_custom.txt \
     --output_path soft_labels_custom
   ```

3. **Update Configuration:**
   ```yaml
   # distillation/configs/distillation_config.yaml
   dataset:
     soft_labels_path: "soft_labels_custom"
     allowed_bird_classes: ["Poecile_montanus", "Certhia_familiaris", "Apus_apus", "Bubo_bubo"]
   ```

4. **Train:**
   ```bash
   # Standard distillation
   python train_distillation.py --config-name=distillation_config
   
   # Or with focal loss for imbalanced custom species
   python train_distillation.py --config-name=focal_loss_config
   ```

### Available Configuration Files

The distillation system includes eight comprehensive configuration files:

| Configuration File | Purpose | Use Case |
|-------------------|---------|----------|
| `distillation_config.yaml` | Standard knowledge distillation | Balanced datasets with teacher model |
| `focal_loss_config.yaml` | Focal + distillation | Imbalanced data with teacher model |
| `pure_focal_config.yaml` | Pure focal loss | Imbalanced data without teacher model |
| `adaptive_focal_config.yaml` | Adaptive focal | Severe imbalance (1:100+ ratio) |
| `manual_weights_config.yaml` | Manual class weights | Custom weight specification |
| `test_distillation.yaml` | Testing configuration | Quick testing and validation |
| `test_split_fix.yaml` | Test with focal | Development with focal distillation |

## Advanced Loss Functions: Focal Loss

### 🎯 **Overview**

The **Focal Loss** implementation addresses the critical problem of **class imbalance** in bird sound classification, where some species are much rarer than others. It focuses training on hard-to-classify examples while down-weighting easy examples.

### ⚙️ **Configurable Parameters**

#### **1. Loss Configuration**

```yaml
loss:
  type: "focal_distillation"    # Loss type
  gamma: 2.0                   # Focusing parameter (0=CE, 2=standard, 3+=strong focus)
  class_weights: "auto"        # Weight calculation method
  alpha_scaling: 1.0           # Weight scaling factor
  
  # FAST WEIGHT CALCULATION PARAMETERS
  use_fast_sampling: true      # Use statistical sampling instead of full scan
  weight_calculation_samples: 500  # Number of samples for weight calculation
  cache_max_age_hours: 24      # Cache validity duration
```

#### **2. Distillation Parameters**
```yaml
distillation:
  alpha: 0.3                   # Balance: (1-α)*hard + α*soft loss
  temperature: 4.0             # Softmax temperature for knowledge transfer
  adaptive: false              # Enable adaptive alpha adjustment
  confidence_threshold: 0.05   # Minimum teacher confidence
```

### 📋 **Available Configurations**

| Configuration | Purpose | Best For | Key Features |
|---------------|---------|----------|--------------|
| **`focal_loss_config.yaml`** | Standard focal+distillation | Imbalanced data + teacher model | Auto weights, 500 samples |
| **`manual_weights_quick.yaml`** | Instant testing | Ultra-fast iteration | Pre-defined weights, no calculation |
| **`pure_focal_config.yaml`** | Pure focal loss | Imbalanced data, no teacher | No distillation, direct training |
| **`adaptive_focal_config.yaml`** | Severe imbalance | Extreme class ratios (1:100+) | Auto-adjusting gamma |

### 🚀 **Quick Start Guide**

#### **Ultra-Fast Testing (Instant Startup)**
```bash
# Use manual weights - NO weight calculation needed
./run_docker_distillation.sh test_instant MAC --config-name=manual_weights_quick

# Startup time: <5 seconds
# Perfect for rapid iteration and debugging
```

#### **Fast Automatic Weights (Recommended)**
```bash
# Uses 500 samples for weight calculation (~30 seconds)
./run_docker_distillation.sh my_training MAC --config-name=focal_loss_config

# Good balance of accuracy and speed
```

#### **Production Training**
```bash
# Full dataset analysis for optimal weights
./run_docker_distillation.sh production MAC --config-name=focal_loss_config \
  loss.weight_calculation_samples=2000 \
  training.epochs=50
```

### ⚡ **Performance Optimizations**

#### **1. Fast Weight Calculation**
- **Before**: Scanned entire dataset (could take hours)
- **Now**: Statistical sampling with 100-2000 samples (10-100x faster)
- **Cache**: Results saved for 24 hours, reused instantly

#### **2. Manual Weight Options**
```yaml
# Skip calculation entirely with manual weights
loss:
  type: "focal_distillation"
  gamma: 2.0
  class_weights: [1.5, 1.2, 0.8]  # [class_0, class_1, class_2, ...]
```

#### **3. Configuration Speed Comparison**

| Configuration | Weight Calculation | Startup Time | Use Case |
|---------------|-------------------|--------------|----------|
| Manual weights | **None** | **<5 seconds** | Development, debugging |
| Fast sampling (500) | Statistical | ~30 seconds | Standard training |
| Fast sampling (100) | Statistical | ~10 seconds | Quick experiments |
| Full dataset | Complete scan | 10+ minutes | Production optimization |

### 🎨 **Parameter Guidelines**

#### **Gamma (Focusing Parameter)**
- `γ = 0`: Standard cross-entropy (no focusing)
- `γ = 1`: Mild focusing on hard examples
- `γ = 2`: **Standard focal loss** (recommended starting point)
- `γ = 3+`: Strong focusing (for severe imbalance)

#### **Class Weight Strategies**
```yaml
# Automatic calculation (recommended)
class_weights: "auto"

# No weighting (equal importance)
class_weights: null

# Manual specification
class_weights: [1.5, 1.0, 2.0, 0.8]  # Per-class weights
```

#### **Alpha Scaling**
- `1.0`: Standard inverse frequency weighting
- `< 1.0`: Reduced class differences
- `> 1.0`: Enhanced class differences

### 🔧 **Advanced Features**

#### **1. Cache Management**
```yaml
loss:
  cache_max_age_hours: 168    # 1 week cache
  force_recalculate: false    # Override cache
```

#### **2. Sampling Control**
```yaml
loss:
  use_fast_sampling: true
  weight_calculation_samples: 1000  # Adjust based on dataset size
  sampling_strategy: "stratified"   # Maintain class balance
```

#### **3. Debug and Monitoring**
```yaml
training:
  log_class_weights: true     # Print computed weights
  save_weight_cache: true     # Save for inspection
```

### 📊 **Performance Impact**

**Training Speed Improvements:**
- **Instant startup**: Manual weights configuration
- **30x faster**: Statistical sampling vs full dataset scan
- **Maintained accuracy**: No performance degradation with fast sampling

**Memory Efficiency:**
- **Reduced computation**: Only processes needed samples
- **Smart caching**: Avoids redundant calculations
- **Configurable limits**: Adapts to available memory

### 🛠️ **Troubleshooting**

#### **Common Issues**

| Problem | Solution |
|---------|----------|
| "Slow startup" | Use `manual_weights_quick.yaml` |
| "Out of memory during weight calculation" | Reduce `weight_calculation_samples` to 100-200 |
| "Cache not working" | Check `cache_max_age_hours` setting |
| "Poor performance" | Increase `weight_calculation_samples` or use manual weights |

#### **Debugging Commands**
```bash
# Test with minimal samples
./run_docker_distillation.sh debug MAC --config-name=focal_loss_config \
  loss.weight_calculation_samples=50 \
  training.epochs=1

# Force weight recalculation
./run_docker_distillation.sh recalc MAC --config-name=focal_loss_config \
  loss.force_recalculate=true
```

### 🎯 **When to Use Each Configuration**

#### **Development & Testing**
```bash
# Ultra-fast iteration
--config-name=manual_weights_quick
```

#### **Standard Training**
```bash
# Balanced speed and accuracy
--config-name=focal_loss_config
```

#### **Production Deployment**
```bash
# Optimal performance
--config-name=focal_loss_config loss.weight_calculation_samples=2000
```

#### **Severe Class Imbalance**
```bash
# Extreme imbalance handling
--config-name=adaptive_focal_config
```

This comprehensive focal loss implementation ensures that class imbalance is handled effectively while maintaining training efficiency and providing flexible configuration options for different use cases.

## Model Benchmarking System

Comprehensive system to compare your trained model with BirdNET as a reference.

### What the Benchmark Does

1. **Audio Discovery**: Automatically scans your dataset and generates ground truth
2. **Student Predictions**: Generates predictions using your trained model  
3. **BirdNET Predictions**: Generates reference predictions using BirdNET
4. **Performance Analysis**: Creates detailed metrics, visualizations, and reports

### Quick Start Examples

#### Docker Execution (Recommended)

**Quick Test (Development):**
```bash
# Test with 10 files to verify everything works
./run_docker_benchmark.sh my_test_gpu1 1 debug.files_limit=10

# Test with small subset
./run_docker_benchmark.sh my_test_gpu1 1 debug.test_with_subset=true
```

**Production Benchmarks:**
```bash
# Full benchmark (may take hours!)
./run_docker_benchmark.sh my_benchmark_gpu1 1 debug.dev_mode=false

# Manageable subset (recommended)
./run_docker_benchmark.sh my_benchmark_gpu1 1 debug.files_limit=1000
```

**Custom Configurations:**
```bash
# Use specific model
./run_docker_benchmark.sh my_benchmark_gpu1 1 \
  benchmark.paths.student_model=my_custom_model.pt

# Adjust confidence thresholds
./run_docker_benchmark.sh my_benchmark_gpu1 1 \
  student_model.inference.confidence_threshold=0.2 \
  birdnet.confidence_threshold=0.1

# Multiple parameters
./run_docker_benchmark.sh my_benchmark_gpu1 1 \
  benchmark.paths.student_model=custom_model.pt \
  debug.files_limit=500 \
  student_model.inference.confidence_threshold=0.15 \
  comparison.save_plots=true
```

#### Local Environment Execution
```bash
cd benchmark
source ../venv/bin/activate
python run_benchmark.py --config-name=quick_start

# With overrides
python run_benchmark.py --config-name=quick_start \
  debug.files_limit=100 \
  benchmark.paths.student_model=../my_model.pt
```

### Benchmark Results

All results are automatically saved in `benchmark/benchmark_results/`:

```
benchmark_results/
├── predictions/                    # Raw prediction files
│   ├── ground_truth.csv           # Auto-generated ground truth
│   ├── student_predictions.csv    # Your model's predictions  
│   └── birdnet_predictions.csv    # BirdNET reference predictions
│
├── comparison/                     # Analysis and visualizations
│   ├── comparison_report.txt      # Human-readable summary
│   ├── comparison_report.json     # Complete metrics in JSON
│   ├── detailed_cases.csv         # Per-file prediction details
│   ├── confusion_matrices.png     # Side-by-side confusion matrices
│   ├── agreement_analysis.png     # Model agreement visualization
│   ├── per_class_accuracy.png     # Per-species accuracy comparison
│   ├── metrics_comparison_table.csv  # Overall metrics comparison
│   └── per_class_metrics_table.csv   # Detailed per-class metrics
│
└── hydra_outputs/                  # Execution logs and configs
    └── [timestamp]/
        ├── main.log               # Complete execution log
        └── .hydra/
            ├── config.yaml        # Final configuration used
            └── overrides.yaml     # Parameters overridden
```

### Configuration Options

#### Quick Test Configuration
```bash
# Minimal test with 3 files
debug.test_with_subset=true debug.subset_size=3

# Test with custom file limit
debug.files_limit=50

# Use different model
benchmark.paths.student_model=path/to/your/model.pt

# Adjust confidence thresholds
student_model.inference.confidence_threshold=0.1
birdnet.confidence_threshold=0.1
```

#### Production Configuration
```bash
# Full dataset evaluation
debug.dev_mode=false debug.files_limit=null

# High-confidence predictions only
student_model.inference.confidence_threshold=0.5
birdnet.confidence_threshold=0.3

# Custom output directory
benchmark.paths.output_dir=results_high_confidence

# Performance optimization
student_model.inference.batch_size=32
```

### Understanding the Metrics

The benchmark provides comprehensive evaluation:

- **Overall Accuracy**: Total correct predictions / total predictions
- **Per-Class Metrics**: Precision, recall, F1-score for each bird species
- **Confusion Matrices**: Visual representation of classification errors
- **Agreement Analysis**: 
  - Both models correct
  - Only student correct  
  - Only BirdNET correct
  - Both models incorrect
- **Confidence Distributions**: Model confidence in predictions
- **Error Analysis**: Detailed breakdown of misclassifications

### Customization Examples

#### Adding New Species
1. Update training config: `config/bird_classification.yaml`
2. Retrain your model with new species
3. Benchmark automatically detects new classes

#### Performance Optimization
```bash
# Faster testing with file limits
debug.files_limit=100

# Memory optimization
student_model.inference.batch_size=16

# Skip visualization for speed
comparison.save_plots=false
```

#### Advanced Analysis
```bash
# Very low confidence threshold (catch more predictions)
student_model.inference.confidence_threshold=0.01
birdnet.confidence_threshold=0.01

# Focus on high-confidence analysis
student_model.inference.confidence_threshold=0.8
birdnet.confidence_threshold=0.5
```

### Troubleshooting

**Common Issues:**

| Problem | Solution |
|---------|----------|
| "No audio files found" | Verify `bird_sound_dataset/` and `augmented_dataset/no_birds/` exist |
| "Model loading failed" | Check model path in config: `benchmark.paths.student_model` |
| "BirdNET species not found" | Some species may not be in BirdNET's database |
| GPU memory issues | Reduce batch size: `student_model.inference.batch_size=8` |
| Docker permission issues | Ensure user has Docker access and GPU permissions |
| **"LexerNoViableAltException: 1"** | **Use correct script syntax: `GPU_ID=1` not just `1`** |
| **"ModuleNotFoundError: hydra"** | **Use Docker scripts or activate virtual environment** |

**Docker Script Syntax Reference:**
```bash
# ✅ CORRECT - All scripts except benchmark
./run_docker_training.sh container_name GPU_ID=1 [hydra_overrides...]
./run_docker_distillation.sh container_name GPU_ID=1 [hydra_overrides...]
./run_docker_soft_labels.sh container_name GPU_ID=1 [additional_args...]

# ✅ CORRECT - Benchmark script (different syntax)
./run_docker_benchmark.sh container_name 1 [hydra_overrides...]

# ❌ WRONG - This causes Hydra parsing errors
./run_docker_distillation.sh container_name 1  # Don't do this!
```

**Performance Tips:**
- Use `debug.files_limit=100` for rapid iteration
- BirdNET is slower than student models - consider subsets for quick tests
- Results are cached to avoid recomputation
- Use Docker for consistent environment across different machines

## Docker Execution

Containerized execution for consistent, reproducible training and benchmarking.

### 1. Build Docker Images
```bash
# Main training image
docker build -t bird_classification_edge .

# Benchmark-specific image
docker build -f Dockerfile.benchmark -t bird_classification_benchmark .
```

### 2. Available Docker Scripts

| Script | Purpose | Example Usage |
|--------|---------|---------------|
| `run_docker_soft_labels.sh` | Extract BirdNET soft labels | `./run_docker_soft_labels.sh my_extraction GPU_ID=0` |
| `run_docker_distillation.sh` | Knowledge distillation training | `./run_docker_distillation.sh my_training GPU_ID=0` |
| `run_docker_benchmark.sh` | **Model benchmarking** | `./run_docker_benchmark.sh my_benchmark 1` |
| `run_docker_training.sh` | Standard training | `./run_docker_training.sh my_training GPU_ID=0` |

### 3. Docker Workflow Examples

#### Complete Knowledge Distillation Pipeline
    ```bash
# Step 1: Extract soft labels from BirdNET
./run_docker_soft_labels.sh extraction_gpu0 GPU_ID=0
# Results saved to soft_labels_complete/

# Step 2: Train with knowledge distillation
./run_docker_distillation.sh training_gpu0 GPU_ID=0 training.epochs=50
# Model saved as best_distillation_model.pt

# Step 3: Benchmark against BirdNET
./run_docker_benchmark.sh benchmark_gpu1 1 debug.files_limit=1000
# Results in benchmark/benchmark_results/
```

#### GPU Management
    ```bash
# Use specific GPU
./run_docker_benchmark.sh my_benchmark 2  # Uses GPU 2
./run_docker_training.sh my_training GPU_ID=2  # Uses GPU 2
./run_docker_distillation.sh my_distill GPU_ID=2  # Uses GPU 2

# CPU-only execution (Mac/no GPU)
./run_docker_soft_labels.sh my_extraction MAC  # Special MAC flag
./run_docker_training.sh my_training MAC  # CPU-only training
```

#### Parameter Overrides
    ```bash
# Training parameters
./run_docker_distillation.sh my_training GPU_ID=0 \
  training.epochs=100 \
  training.batch_size=64 \
  optimizer.lr=0.0005

# Standard training parameters
./run_docker_training.sh my_training GPU_ID=0 \
  training.epochs=50 \
  optimizer.lr=0.001

# Benchmark parameters  
./run_docker_benchmark.sh my_benchmark 1 \
  debug.files_limit=500 \
  student_model.inference.confidence_threshold=0.2 \
  benchmark.paths.student_model=custom_model.pt
```

## Configuration Reference

### Main Training Configuration (`config/bird_classification.yaml`)

#### Dataset Parameters
    ```yaml
    dataset:
  bird_data_dir: "bird_sound_dataset"
  allowed_bird_classes: ["Bubo_bubo", "Apus_apus", "Certhia_familiaris", "Poecile_montanus"]
  
  # "No Birds" class handling
  load_pregenerated_no_birds: true
  pregenerated_no_birds_dir: "augmented_dataset/no_birds/"
  num_no_bird_samples: 100
  esc50_no_bird_ratio: 0.5
  
  # Audio processing
  target_sr: 22050
  clip_duration: 3.0
  extract_calls: true
  
  # Data loading
  num_workers: 4
  split_ratios: [0.7, 0.15, 0.15]  # train/val/test
```

#### Training Parameters
```yaml
training:
  epochs: 100
  batch_size: 32
  patience: 15
  checkpoint_every: 10
  
optimizer:
  type: "Adam"
  lr: 0.001
  weight_decay: 1e-4
  
model:
  n_classes: 5  # 4 bird species + 1 no_birds
  dropout: 0.3
```

#### Augmentation Settings
```yaml
dataset:
  augmentation:
    enabled: true
    noise_level: 0.005
    time_mask_param: 80
    freq_mask_param: 80
    mixup_alpha: 0.2
```

### Benchmark Configuration (`benchmark/config/`)

#### Quick Start Config (`quick_start.yaml`)
```yaml
benchmark:
  paths:
    audio_dir: "../bird_sound_dataset"
    no_birds_dir: "../augmented_dataset/no_birds"
    student_model: "../best_distillation_model.pt"
    student_config: "../config/bird_classification.yaml"
    output_dir: "benchmark_results"

debug:
  dev_mode: true
  files_limit: 100
  test_with_subset: false
  
student_model:
  inference:
    device: "cuda"
    batch_size: 32
    confidence_threshold: 0.1
```

#### Full Benchmark Config (`benchmark.yaml`)
```yaml
benchmark:
  paths:
    audio_dir: "../bird_sound_dataset"
    student_model: "../best_distillation_model.pt"
    output_dir: "benchmark_results"

debug:
  dev_mode: false
  files_limit: null  # No limit
  
comparison:
  save_plots: true
  save_detailed_json: true
  plot_style: "seaborn"
```

### Knowledge Distillation Configuration (`distillation/configs/`)

#### Standard Distillation (`distillation_config.yaml`)
```yaml
training:
  alpha: 0.5        # Balance between hard and soft loss
  temperature: 4.0  # Softmax temperature for distillation
  epochs: 100

loss:
  type: "distillation"  # Standard cross-entropy + distillation
  
dataset:
  soft_labels_path: "soft_labels_complete"
  allowed_bird_classes: ["Bubo_bubo", "Apus_apus", "Certhia_familiaris", "Poecile_montanus"]
```

#### Focal Loss + Distillation (`focal_loss_config.yaml`)
```yaml
training:
  alpha: 0.3        # Balance for focal distillation
  temperature: 4.0
  epochs: 100

loss:
  type: "focal_distillation"  # Focal loss + knowledge distillation
  gamma: 2.0                  # Focusing parameter
  class_weights: "auto"       # Automatic class weight computation
  alpha_scaling: 1.0          # Weight scaling factor

dataset:
  soft_labels_path: "soft_labels_complete"
  allowed_bird_classes: ["Bubo_bubo", "Apus_apus", "Certhia_familiaris", "Poecile_montanus"]
```

#### Pure Focal Loss (`pure_focal_config.yaml`)
```yaml
training:
  epochs: 100

loss:
  type: "focal"             # Pure focal loss, no distillation
  gamma: 2.0
  class_weights: "auto"
  alpha_scaling: 1.0

dataset:
  allowed_bird_classes: ["Bubo_bubo", "Apus_apus", "Certhia_familiaris", "Poecile_montanus"]
  # No soft_labels_path needed
```

### Loss Function Selection Guide

Choose the appropriate loss function configuration based on your dataset characteristics:

#### When to Use Each Configuration

| Data Characteristics | Recommended Config | Key Benefits |
|---------------------|-------------------|--------------|
| **Balanced classes + teacher model** | `distillation_config.yaml` | Standard knowledge transfer |
| **Imbalanced classes + teacher model** | `focal_loss_config.yaml` | Handles imbalance + knowledge transfer |
| **Imbalanced classes, no teacher** | `pure_focal_config.yaml` | Pure focal loss for imbalanced data |
| **Severe imbalance (1:100+ ratio)** | `adaptive_focal_config.yaml` | Automatically adjusts to extreme imbalance |
| **Domain expertise available** | `manual_weights_config.yaml` | Manual control over class weights |

#### Parameter Guidelines

**Gamma (Focusing Parameter):**
- `γ = 0`: Equivalent to cross-entropy (no focusing)
- `γ = 1`: Mild focusing on hard examples
- `γ = 2`: Standard focal loss (recommended starting point)
- `γ = 3+`: Strong focusing (use for severe imbalance)

**Class Weight Options:**
- `null`: Equal weights for all classes
- `"auto"`: Automatically computed from data distribution
- `[1.0, 2.0, ...]`: Manual specification (list of weights per class)

**Alpha Scaling:**
- `1.0`: Standard inverse frequency weighting
- `< 1.0`: Reduced weight differences between classes
- `> 1.0`: Enhanced weight differences between classes

---

## 📚 Additional Resources

- **Benchmark System**: See `benchmark/README.md` for detailed benchmarking guide
- **Knowledge Distillation**: See `distillation/README.md` for distillation specifics
- **Dataset Preparation**: See examples in `datasets/test_datasets.py`
- **Model Architecture**: Detailed in `models.py` and `modules.py`

## 🤝 Contributing

1. Follow the modular structure when adding new features
2. Add comprehensive documentation for new components
3. Include tests in `datasets/test_datasets.py` for new dataset classes
4. Update relevant README files for significant changes

## 📄 License

[Add your license information here]

---

**Need Help?** Check the troubleshooting sections in each component's documentation or review the extensive logging output for debugging information.