# Bird Sound Classification on Edge Devices

This repository contains a comprehensive bird sound classification system designed to run on edge devices. It includes functionality for dataset preparation, augmentation, training, knowledge distillation, and benchmarking.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Standard Training Workflow](#standard-training-workflow)
- [Advanced Workflow: Knowledge Distillation](#advanced-workflow-knowledge-distillation)
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
│   │   ├── losses/                  # Distillation loss functions
│   │   └── configs/                 # Distillation configurations
│   ├── extract_soft_labels.py       # Extract BirdNET soft labels
│   └── train_distillation.py        # Train with knowledge distillation
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
# Use default configuration
python train_distillation.py

# With custom parameters
python train_distillation.py \
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
   python train_distillation.py
   ```

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
| `run_docker_soft_labels.sh` | Extract BirdNET soft labels | `./run_docker_soft_labels.sh my_extraction 0` |
| `run_docker_distillation.sh` | Knowledge distillation training | `./run_docker_distillation.sh my_training 0` |
| `run_docker_benchmark.sh` | **Model benchmarking** | `./run_docker_benchmark.sh my_benchmark 1` |
| `run_docker_training.sh` | Standard training | `./run_docker_training.sh my_training 0` |

### 3. Docker Workflow Examples

#### Complete Knowledge Distillation Pipeline
```bash
# Step 1: Extract soft labels from BirdNET
./run_docker_soft_labels.sh extraction_gpu0 0
# Results saved to soft_labels_complete/

# Step 2: Train with knowledge distillation
./run_docker_distillation.sh training_gpu0 0 training.epochs=50
# Model saved as best_distillation_model.pt

# Step 3: Benchmark against BirdNET
./run_docker_benchmark.sh benchmark_gpu1 1 debug.files_limit=1000
# Results in benchmark/benchmark_results/
```

#### GPU Management
```bash
# Use specific GPU
./run_docker_benchmark.sh my_benchmark 2  # Uses GPU 2

# CPU-only execution (Mac/no GPU)
./run_docker_soft_labels.sh my_extraction MAC  # Special MAC flag
```

#### Parameter Overrides
```bash
# Training parameters
./run_docker_distillation.sh my_training 0 \
  training.epochs=100 \
  training.batch_size=64 \
  optimizer.lr=0.0005

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

```yaml
# distillation_config.yaml
training:
  alpha: 0.5        # Balance between hard and soft loss
  temperature: 4.0  # Softmax temperature for distillation
  epochs: 100
  
dataset:
  soft_labels_path: "soft_labels_complete"
  allowed_bird_classes: ["Bubo_bubo", "Apus_apus", "Certhia_familiaris", "Poecile_montanus"]
```

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