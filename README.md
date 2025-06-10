# Bird Sound Classification on Edge Devices

This repository contains a bird sound classification system designed to run on edge devices. It includes functionality for dataset preparation, augmentation, and training.

## Project Structure

The codebase is organized into a modular structure:

```
.
├── datasets/                # Dataset handling modules
│   ├── __init__.py          # Package initialization
│   ├── audio_utils.py       # Audio processing utilities
│   ├── bird_dataset.py      # Bird sound dataset implementation
│   ├── empty_segment_dataset.py  # Empty/silent segment extraction
│   ├── esc50_dataset.py     # ESC-50 environmental sound dataset
│   ├── dataset_factory.py   # Combined dataset creation
│   └── test_datasets.py     # Testing functionality
├── distillation/            # Knowledge distillation package
│   ├── scripts/             # Executable scripts for distillation
│   │   ├── extract_soft_labels.py  # Extract soft labels from BirdNET
│   │   └── train_distillation.py   # Train with knowledge distillation
│   ├── datasets/            # Dataset classes with soft labels
│   │   └── distillation_dataset.py # Loads both hard and soft labels
│   ├── losses/              # Distillation loss functions
│   │   └── distillation_loss.py    # Combined hard + soft loss
│   ├── configs/             # Distillation configurations
│   │   ├── distillation_config.yaml # Main distillation config
│   │   └── test_distillation.yaml   # Quick test configuration
│   └── README.md            # Distillation package documentation
├── generate_no_birds_samples.py  # Script to generate "no_birds" class samples (offline mode)
├── train.py                 # Main training script
├── extract_soft_labels.py   # Convenience wrapper for distillation extraction
├── train_distillation.py    # Convenience wrapper for distillation training
├── config/                  # Configuration files
│   └── bird_classification.yaml  # Main configuration
└── [other project files]
```

## Knowledge Distillation with BirdNET

This project includes a complete knowledge distillation pipeline using BirdNET as the teacher model to improve the accuracy of lightweight student models suitable for edge deployment.

### Overview

Knowledge distillation is a model compression technique where a small "student" model learns from a larger, more accurate "teacher" model. In our case:
- **Teacher**: BirdNET (large, pre-trained on thousands of bird species)
- **Student**: Our lightweight model (~53k parameters, suitable for AudioMoth)
- **Goal**: Transfer BirdNET's knowledge while maintaining edge deployment compatibility

### 📁 Distillation Package Structure

```
distillation/
├── scripts/                     # Executable scripts
│   ├── extract_soft_labels.py  # Extract soft labels from BirdNET teacher
│   └── train_distillation.py   # Train student with distillation
├── datasets/                    # Dataset classes with soft labels
│   └── distillation_dataset.py # Loads both hard and soft labels
├── losses/                      # Distillation loss functions
│   └── distillation_loss.py    # Combined hard + soft loss
└── configs/                     # Configuration files
    ├── distillation_config.yaml # Main distillation config
    └── test_distillation.yaml   # Quick test configuration
```

### 🔄 Distillation Workflow

#### Step 1: Extract Soft Labels from BirdNET Teacher

```bash
# Extract soft probability distributions from BirdNET for all audio files
python extract_soft_labels.py --confidence_threshold 0.05

# Advanced usage with custom parameters
python extract_soft_labels.py \
  --dataset_path bird_sound_dataset \
  --output_path soft_labels \
  --confidence_threshold 0.03 \
  --max_files_per_class 100  # For testing with subset
```

This script:
- Analyzes each audio file with BirdNET
- Extracts probability distributions across all bird species
- Maps BirdNET's 70 species to your dataset's classes
- Handles "non-bird" classification for environmental sounds
- Saves results as `soft_labels/soft_labels.json` and metadata

#### Step 2: Train Student Model with Knowledge Distillation

```bash
# Train with distillation using default configuration
python train_distillation.py

# Quick test with subset (3 epochs, 4 bird species)
python train_distillation.py --config-name=test_distillation

# Custom parameters via Hydra overrides
python train_distillation.py \
  distillation.alpha=0.3 \
  distillation.temperature=4.0 \
  training.epochs=30 \
  training.batch_size=16
```

### 🎯 Key Distillation Components

#### DistillationLoss
Combines hard ground truth labels with soft teacher predictions:
```python
# L_total = (1-α) * L_hard + α * L_soft
# L_hard: Standard cross-entropy with ground truth
# L_soft: KL divergence between student and teacher predictions
total_loss = (1 - alpha) * hard_loss + alpha * soft_loss
```

**Key Parameters:**
- `alpha` (0.0-1.0): Balance between hard and soft supervision
  - 0.0 = Standard training (only ground truth)
  - 0.5 = Equal weight to both
  - 1.0 = Pure distillation (only teacher)
- `temperature`: Softens probability distributions for better knowledge transfer

#### Adaptive Distillation
Automatically adjusts the hard/soft balance based on validation performance:
- Increases teacher influence when validation accuracy plateaus
- Helps overcome training difficulties with challenging species

### 📊 Expected Benefits

Based on knowledge distillation literature and our model architecture:

- **Accuracy Improvement**: +2-5% absolute improvement expected
- **Challenging Species**: Better performance on difficult classes (e.g., Poecile montanus)
- **Generalization**: Improved robustness to acoustic variations
- **Inter-class Relationships**: Student learns which species are acoustically similar
- **Edge Compatibility**: Student model size remains unchanged (~53k parameters)

### 🔧 Configuration & Hyperparameters

#### Key Distillation Parameters (`distillation_config.yaml`)

```yaml
distillation:
  alpha: 0.3                    # Teacher influence (start with 0.3)
  temperature: 4.0              # Probability smoothing (2.0-8.0)
  adaptive: false               # Auto-adjust alpha based on validation
  confidence_threshold: 0.05    # Min confidence for teacher predictions

training:
  lr: 0.0005                   # Often needs lower LR than standard training
  epochs: 30                   # More epochs beneficial for distillation
  batch_size: 16               # Can be larger since teacher runs offline
```

#### Hyperparameter Guidelines

| Parameter | Range | Description | Tuning Tips |
|-----------|-------|-------------|-------------|
| `alpha` | 0.1-0.7 | Teacher influence weight | Start 0.3, increase if student struggles |
| `temperature` | 1.0-8.0 | Softness of distributions | Higher for more teacher knowledge transfer |
| `learning_rate` | 0.0001-0.001 | Student model learning rate | Often 2-5x lower than standard training |
| `confidence_threshold` | 0.03-0.1 | Min teacher confidence | Lower captures more inter-class relationships |

### 📈 Monitoring Training

The distillation training provides detailed logging:

```
Epoch 1: Train Loss: 1.234, Train Acc: 75.2%, Val Loss: 1.156, Val Acc: 78.1%
  Hard Loss: 1.456, Soft Loss: 0.789, Alpha: 0.300
New best model saved! Val Acc: 78.1%
```

**Key Metrics:**
- **Total Loss**: Combined objective being optimized
- **Hard Loss**: Standard classification performance
- **Soft Loss**: How well student matches teacher distributions
- **Alpha**: Current hard/soft balance (adaptive distillation)

### 🚀 Quick Start Example

```bash
# 1. Extract soft labels (run once per dataset)
python extract_soft_labels.py --confidence_threshold 0.05

# 2. Run quick test (3 epochs, 4 species)
python train_distillation.py --config-name=test_distillation

# 3. Full training run
python train_distillation.py \
  distillation.alpha=0.3 \
  training.epochs=50 \
  training.batch_size=16
```

### 🔬 Advanced Usage

#### Custom Soft Label Extraction
```python
from distillation import DistillationBirdSoundDataset

# Load dataset with soft labels
dataset = DistillationBirdSoundDataset(
    soft_labels_path="soft_labels",
    root_dir="bird_sound_dataset", 
    subset="training"
)

# Access both hard and soft labels
audio, hard_label, soft_labels = dataset[0]
print(f"Hard label: {hard_label}")
print(f"Soft labels shape: {soft_labels.shape}")  # [num_classes]
```

#### Custom Distillation Training Loop
```python
from distillation import DistillationLoss

# Create distillation loss
criterion = DistillationLoss(alpha=0.3, temperature=4.0)

# Training step
student_logits = model(audio)
total_loss, hard_loss, soft_loss = criterion(
    student_logits, hard_labels, teacher_soft_labels
)

# Monitor individual loss components
print(f"Total: {total_loss:.3f}, Hard: {hard_loss:.3f}, Soft: {soft_loss:.3f}")
```

### 📋 Troubleshooting

**Common Issues:**

1. **"No soft labels found" warnings**: 
   - Check that `extract_soft_labels.py` completed successfully
   - Verify `soft_labels/` directory exists with `.json` files

2. **High soft loss, low accuracy**:
   - Try lower `alpha` (0.1-0.2) for more ground truth supervision
   - Reduce `temperature` (2.0-3.0) for sharper teacher distributions

3. **Training instability**:
   - Lower learning rate (0.0001-0.0005)
   - Enable adaptive distillation (`distillation.adaptive: true`)

4. **Out of memory errors**:
   - Reduce `batch_size` in distillation config
   - Soft labels add minimal memory overhead

For detailed documentation on distillation components, see [`distillation/README.md`](distillation/README.md).

## Data Augmentation and "No Birds" Class Workflow

The system supports two main approaches for handling the "no_birds" class samples, controlled via `config/bird_classification.yaml`:

### Mode 1: Offline Generation of "No Birds" Samples (Recommended for Reproducibility and Speed)

1.  **Generate "No Birds" Class Samples (Offline):**
    This step uses `generate_no_birds_samples.py` to create and save audio files for the "no_birds" class.
    ```bash
    python generate_no_birds_samples.py --bird_dir path/to/birds --output_dir augmented_dataset/no_birds --energy_threshold 2.0
    ```
    This script:
    - Creates a balanced set of "no_birds" samples from two sources:
        - **ESC-50 environmental sounds** (rain, footsteps, engines, etc.)
        - **Empty/silent segments** extracted from your bird recordings
    - Saves these to the specified output directory (e.g., `augmented_dataset/no_birds/`).

2.  **Configure Training to Load Pre-generated Samples:**
    In `config/bird_classification.yaml`, set:
    ```yaml
    dataset:
      load_pregenerated_no_birds: true
      pregenerated_no_birds_dir: "augmented_dataset/no_birds/" # Or your custom path
      # num_no_bird_samples: 100 # Optionally, specify how many to load if fewer than available
    ```

### Mode 2: On-the-Fly Generation of "No Birds" Samples

1.  **Configure Training for On-the-Fly Generation:**
    In `config/bird_classification.yaml`, set:
    ```yaml
    dataset:
      load_pregenerated_no_birds: false
      num_no_bird_samples: 100 # Desired total number of "no_birds" samples per epoch/dataset creation
      esc50_no_bird_ratio: 0.5   # Proportion from ESC-50 vs. empty segments
    ```
    In this mode, `train.py` (via `datasets/dataset_factory.py`) will dynamically generate "no_birds" samples during dataset setup using ESC-50 sounds and by extracting silent segments from your bird recordings. No prior execution of `generate_no_birds_samples.py` is strictly needed for the "no_birds" class itself, though you still need your base bird recordings and the ESC-50 dataset.

### Dataset Structure (Example after Offline Generation)

```
bird_sound_dataset/         # Original dataset
  ├── species1/             # Bird class folders
  │   ├── recording1.wav
  │   └── ...
  └── species2/
      └── ...

augmented_dataset/          # Directory for pre-generated data (if using offline mode)
  └── no_birds/             # The "no birds" class samples
      ├── esc50_0000.wav    # Environmental sounds
      ├── ...
      ├── empty_0000.wav    # Silent segments
      └── ...
```

### Common Augmentations Applied During Training (Both Modes)

Once the base bird sounds and "no_birds" samples (either loaded or generated on-the-fly) are ready, `train.py` applies further augmentations in real-time if enabled in the configuration (`dataset.augmentation.enabled: true`):
- Bird call extraction (if `dataset.extract_calls: true`)
- Background mixing with controlled SNR (using ESC-50 sounds as background)
- Time/frequency masking
- Speed perturbation
- Time shifting

## Key Components

### `generate_no_birds_samples.py`
Script to generate and save "no_birds" class samples offline. Useful for creating a fixed set of negative samples.

### `datasets/dataset_factory.py` (`create_combined_dataset` function)
This is the core function called by `train.py` to prepare the training, validation, and test datasets. It now supports:
- Loading pre-generated "no_birds" samples from disk.
- Generating "no_birds" samples on-the-fly from ESC-50 and silent segments of bird recordings.
The choice is controlled by `load_pregenerated_no_birds` in the configuration.

### `datasets/empty_segment_dataset.py` (`EmptySegmentDataset` class)
A dataset class that finds and extracts segments with low energy (likely no bird calls) from audio recordings. Used by both `generate_no_birds_samples.py` (offline) and `dataset_factory.py` (if generating on-the-fly).

## Key Configuration Parameters (`config/bird_classification.yaml`)

Below are some important parameters in `config/bird_classification.yaml` related to dataset handling and the "no_birds" class:

```yaml
dataset:
  # Bird sound dataset parameters
  bird_data_dir: "bird_sound_dataset"  # Path to bird sound data
  esc50_dir: "ESC-50-master"           # Path to ESC-50 dataset
  
  # "No Birds" class handling
  load_pregenerated_no_birds: false    # true: Load from disk; false: Generate on-the-fly
  pregenerated_no_birds_dir: "augmented_dataset/no_birds/" # Path if loading from disk
  num_no_bird_samples: 100             # Target number for "no_birds" samples (used in both modes)
                                       # For pre-generated, if available samples < this, all are used.
                                       # If 0 when loading pre-generated, all available samples are used.
  esc50_no_bird_ratio: 0.5             # Proportion from ESC-50 (vs. empty) if generating on-the-fly

  # Augmentation parameters (applied in real-time by BirdSoundDataset)
  augmentation:
    enabled: true                      # Whether to use real-time augmentation
    # ... other augmentation params (noise_level, time_mask_param, etc.)

  # Bird call extraction parameters
  extract_calls: true                  # Whether to extract bird calls from audio
  # ... other params (min_peak_distance, etc.)
```

## Script Parameters

### `generate_no_birds_samples.py` (for Offline Mode)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bird_dir` | bird_sound_dataset | Directory containing bird sound recordings |
| `--esc50_dir` | *auto-downloaded* | Directory containing ESC-50 dataset |
| `--output_dir` | augmented_dataset | Directory to save generated samples |
| `--num_samples` | 100 | Number of "no birds" samples to generate |
| `--esc50_ratio` | 0.5 | Proportion of samples from ESC-50 vs. empty segments |
| `--files_per_class` | None | Maximum files to scan per bird class |
| `--energy_threshold` | 1.5 | Threshold factor for detecting silence (higher = more segments) |

## Important Parameters

- **energy_threshold_factor**: Controls how "silent" segments must be to be included
  - **0.5**: Very strict (original setting) - few segments found
  - **1.5-3.0**: More lenient - finds more segments
  - Values above 3.0 may include segments with actual bird calls

- **SNR (Signal-to-Noise Ratio)** for background mixing: 
  - **5-15 dB**: Default range
  - Lower values = background sounds more prominent

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd bird_classification_edge
   ```

2. Create and Activate a Virtual Environment (Recommended):
   It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects or your system's Python installation.

   *   **Create the virtual environment (e.g., named `venv`):
       ```bash
       python3 -m venv venv
       ```

   *   **Activate the virtual environment:**
       *   On macOS/Linux (bash/zsh):
           ```bash
           source venv/bin/activate
           ```
       *   On Windows (Command Prompt):
           ```bash
           venv\Scripts\activate.bat
           ```
       *   On Windows (PowerShell):
           ```bash
           venv\Scripts\Activate.ps1
           ```
       (Your terminal prompt should change to indicate the active environment, e.g., `(venv)`).

3. Install dependencies:
   Once the virtual environment is activated, install the required packages using the `requirements.txt` file (if available and up-to-date) or install them manually.
   ```bash
   pip install -r requirements.txt
   ```
   If `requirements.txt` is missing or outdated, you can install common packages like:
   ```bash
   pip install torch torchaudio torchvision numpy pandas matplotlib seaborn scikit-learn hydra-core omegaconf tqdm
   ```

4. Download bird recordings or use your own dataset. Ensure your bird sounds are organized into subdirectories by species within a main data directory (e.g., `bird_sound_dataset/species_A/sound1.wav`).

5. **Decide on "No Birds" Sample Strategy:**
    a. **Offline Mode (Recommended Start):** Run `generate_no_birds_samples.py` to create them (e.g., into `augmented_dataset/no_birds/`). Then, ensure `load_pregenerated_no_birds: true` and `pregenerated_no_birds_dir` is correctly set in your `config/bird_classification.yaml`.
    b. **On-the-Fly Mode:** Ensure `load_pregenerated_no_birds: false` in your config. The script will generate them during setup using ESC-50 and silent segments from your bird data.

6. Configure training parameters in `config/bird_classification.yaml` (especially the `dataset` section reflecting your choice above, and paths like `bird_data_dir`, `esc50_dir`).

7. Run training:
   ```bash
   python train.py # Add any Hydra command-line overrides if needed
   ```

## Docker Deployment for Knowledge Distillation

The knowledge distillation pipeline can be run entirely in Docker containers, making it ideal for server deployment with GPU support.

### Prerequisites

- Docker installed with GPU support (nvidia-docker2 for Linux servers)
- Required datasets mounted in the project directory:
  - `bird_sound_dataset/` - Bird audio files  
  - `augmented_dataset/no_birds/` - Non-bird samples
  - `esc-50/ESC-50-master/` - ESC-50 environmental sounds

### Build Docker Image

```bash
# Build the Docker image with all dependencies
docker build -t bird_classification_edge .
```

### Knowledge Distillation Workflow

#### Step 1: Extract Soft Labels from BirdNET Teacher

```bash
# Basic usage - extract soft labels for all files
./run_docker_soft_labels.sh leonardo_soft_labels_gpu0 GPU_ID=0

# With custom parameters
./run_docker_soft_labels.sh leonardo_extraction GPU_ID=1 --confidence_threshold 0.03 --max_files_per_class 500

# For macOS (CPU only)
./run_docker_soft_labels.sh leonardo_test_mac MAC
```

**What this does:**
- Runs BirdNET teacher model on all audio files (~5000 files)
- Generates probability distributions for each file
- Saves to `soft_labels_complete/` directory
- Expected runtime: 2-3 hours on GPU server

#### Step 2: Train Student with Knowledge Distillation

```bash
# Basic distillation training
./run_docker_distillation.sh leonardo_distillation_gpu0 GPU_ID=0

# Full training with custom hyperparameters
./run_docker_distillation.sh leonardo_kd_training GPU_ID=1 \
  distillation.alpha=0.3 \
  distillation.temperature=4.0 \
  training.epochs=50 \
  training.batch_size=16

# Quick test run
./run_docker_distillation.sh leonardo_test GPU_ID=0 \
  --config-name=test_distillation \
  training.epochs=3
```

**What this does:**
- Trains student model using both hard labels and BirdNET soft labels
- Uses 5-class classification (4 birds + non-bird)
- Automatically uses complete soft labels from Step 1
- Expected runtime: 2-4 hours depending on epochs

### GPU Management on Server

```bash
# Check GPU usage
watch nvidia-smi

# Use specific GPU (recommended on shared servers)
./run_docker_soft_labels.sh your_unique_name GPU_ID=1
./run_docker_distillation.sh your_unique_name GPU_ID=1

# Check running containers
docker ps
```

### Container Naming Convention

Use descriptive, unique container names on shared servers:
- `leonardo_extraction_gpu0` - Soft label extraction
- `leonardo_distillation_gpu1` - Distillation training  
- `teamname_purpose_gpu#` - General pattern

### Directory Structure (Server)

```
/path/to/project/
├── bird_sound_dataset/          # ~4240 bird audio files
├── augmented_dataset/no_birds/  # ~836 non-bird samples  
├── esc-50/ESC-50-master/       # ESC-50 environmental sounds
├── soft_labels_complete/        # Generated by extraction (Step 1)
│   ├── soft_labels.json         # Probability distributions
│   └── soft_labels_metadata.json # Teacher model info
├── logs/                        # Training logs and models
└── [scripts and configs]
```

### Monitoring and Troubleshooting

#### Check Extraction Progress
```bash
# Monitor soft labels extraction
docker logs leonardo_extraction_gpu0

# Check number of files processed
ls soft_labels_complete/ && wc -l soft_labels_complete/soft_labels.json
```

#### Check Training Progress  
```bash
# Monitor training logs
docker logs leonardo_distillation_gpu0

# Check latest model checkpoint
ls -la logs/bird_classification_distillation/
```

#### Common Issues

1. **"Directory not found" errors:**
   ```bash
   # Verify all required directories exist
   ls -la bird_sound_dataset/ augmented_dataset/no_birds/ esc-50/ESC-50-master/
   ```

2. **"Soft labels file not found":**
   - Run Step 1 (extraction) first before Step 2 (training)
   - Check that `soft_labels_complete/soft_labels.json` exists

3. **GPU out of memory:**
   ```bash
   # Reduce batch size for distillation
   ./run_docker_distillation.sh name GPU_ID=0 training.batch_size=8
   ```

4. **Container name conflicts:**
   ```bash
   # Use unique names or remove existing container
   docker rm old_container_name
   ```

### Expected Results

After successful knowledge distillation:
- **Accuracy improvement**: +2-5% over baseline student model
- **Model size**: Still ~53k parameters (edge-compatible)
- **Better generalization**: Especially for challenging species
- **Robustness**: Improved performance on acoustic variations

### Advanced Docker Usage

#### Custom Soft Label Extraction
```bash
# Extract subset for testing
./run_docker_soft_labels.sh test_extraction GPU_ID=0 \
  --max_files_per_class 10 \
  --confidence_threshold 0.1

# Extract with different output directory  
./run_docker_soft_labels.sh custom_extraction GPU_ID=0 \
  --output_path custom_soft_labels
```

#### Custom Distillation Configuration
```bash
# Use custom config file
./run_docker_distillation.sh custom_training GPU_ID=0 \
  --config-name=custom_distillation_config \
  soft_labels_path=custom_soft_labels

# Adaptive distillation with different alpha schedule
./run_docker_distillation.sh adaptive_training GPU_ID=0 \
  distillation.adaptive=true \
  distillation.adaptation_rate=0.05 \
  distillation.alpha_schedule=cosine
```

#### Background Execution
```bash
# Run extraction in background (for long jobs)
nohup ./run_docker_soft_labels.sh bg_extraction GPU_ID=0 > extraction.log 2>&1 &

# Monitor background job
tail -f extraction.log
```

## Running with Docker
