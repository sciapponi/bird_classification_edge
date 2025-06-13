# Bird Sound Classification on Edge Devices

This repository contains a bird sound classification system designed to run on edge devices. It includes functionality for dataset preparation, augmentation, and training.

## Table of Contents

- [Installation and Setup](#installation-and-setup)
- [Project Structure](#project-structure)
- [Standard Training Workflow](#standard-training-workflow)
- [Advanced Workflow: Knowledge Distillation](#advanced-workflow-knowledge-distillation)
- [Docker Execution](#docker-execution)

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd bird_classification_edge
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Datasets:**
    -   Place your bird recordings in `bird_sound_dataset/`, organized by species folders.
    -   The ESC-50 dataset for environmental sounds will be downloaded automatically on the first run if not found in `esc-50/`.

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

## Standard Training Workflow

This is the primary workflow for training the model without knowledge distillation. It uses the standard dataset and `CrossEntropyLoss`.

### 1. Configure Your Training

-   Edit the main configuration file: `config/bird_classification.yaml`.
-   Adjust parameters like `epochs`, `batch_size`, `learning_rate`.
-   Select the desired bird species under `dataset.allowed_bird_classes`.

### 2. Handle "No Birds" Samples

You can choose one of two modes in your configuration file:

-   **Offline Mode (Recommended):** First, generate a fixed set of "no-bird" samples.
    ```bash
    python generate_no_birds_samples.py
    ```
    Then, in `config/bird_classification.yaml`, set `load_pregenerated_no_birds: true`.

-   **Online Mode:** Set `load_pregenerated_no_birds: false`. "No-bird" samples will be generated on-the-fly during training setup.

### 3. Run the Training Script
```bash
# Run training with the configuration from your .yaml file
python train.py

# Override any parameter from the command line
python train.py training.epochs=50 optimizer.lr=0.001 dataset.num_workers=8
```

## Advanced Workflow: Knowledge Distillation

This workflow uses a large "teacher" model (BirdNET) to train our smaller "student" model, improving its accuracy and generalization.

### How it Works
The student model learns from two sources:
1.  **Hard Labels:** The ground truth (e.g., "This is `Bubo_bubo`").
2.  **Soft Labels:** The teacher's detailed probabilities (e.g., "70% `Bubo_bubo`, 20% `Apus_apus`").

This is controlled by a `DistillationLoss` function: `L_total = (1-α) * L_hard + α * L_soft`.

### Workflow Steps

#### Step 1: Extract Soft Labels from the Teacher
Run the extraction script to have BirdNET analyze your dataset and produce `soft_labels.json`.

```bash
# Extract soft labels for all classes defined in distillation/species.txt
python extract_soft_labels.py --output_path soft_labels_complete
```

#### Step 2: Train the Student with Distillation
Run the distillation training script. It will use the configuration in `distillation/configs/distillation_config.yaml`.

```bash
# Ensure soft_labels_path in the config points to the correct directory
python train_distillation.py
```

### Training on a Custom Subset of Classes

If you want to train on fewer classes, you must generate new soft labels that match.

1.  **Create a Custom Species List:** Create a new file, e.g., `distillation/species_4.txt`, with only the species you want.
    ```
    # distillation/species_4.txt
    Poecile montanus
    Certhia familiaris
    Apus apus
    Bubo bubo
    ```
2.  **Generate Matching Soft Labels:** Run the extraction script pointing to your new list and a new output folder.
    ```bash
    python extract_soft_labels.py \
        --species_list distillation/species_4.txt \
        --output_path soft_labels_4_classes
    ```
3.  **Configure and Train:** Update `distillation/configs/distillation_config.yaml` to point `soft_labels_path` to `soft_labels_4_classes` and list the same 4 species under `allowed_bird_classes`. Then run `python train_distillation.py`.

## Docker Execution

The knowledge distillation pipeline is fully runnable in Docker, ideal for server deployment.

### 1. Build the Docker Image
```bash
docker build -t bird_classification_edge .
```

### 2. Run the Workflow

-   **Step 1: Extract Soft Labels:**
    ```bash
    # Use GPU 0 for extraction
    ./run_docker_soft_labels.sh my_extraction_gpu0 GPU_ID=0
    
    # Run on a Mac (CPU only)
    ./run_docker_soft_labels.sh my_extraction_mac MAC
    ```
    This saves results to `soft_labels_complete/`.

-   **Step 2: Run Distillation Training:**
    ```bash
    # Use GPU 0 for training
    ./run_docker_distillation.sh my_training_gpu0 GPU_ID=0

    # Override parameters. Note: For distillation, some top-level params like batch_size
    # are in the 'training' block, while dataset params are in the 'dataset' block.
    ./run_docker_distillation.sh my_training_run GPU_ID=1 training.epochs=50 dataset.num_workers=8
    ```
For more details on server usage, GPU management, and troubleshooting, see the comments within the `run_docker_*.sh` scripts.

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
  # General dataset settings
  num_workers: 4                       # Number of parallel data loading processes. Increase to use more CPU cores and speed up data preparation.
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
| `--files_per_class` | None | Maximum files to scan per bird class (for testing) |
| `--energy_threshold`| 1.5 | Energy threshold factor for detecting silence (higher = more segments) |

## Docker Execution

The knowledge distillation pipeline is fully runnable in Docker, ideal for server deployment.

### 1. Build the Docker Image
```bash
docker build -t bird_classification_edge .
```

### 2. Run the Workflow

-   **Step 1: Extract Soft Labels:**
    ```bash
    # Use GPU 0 for extraction
    ./run_docker_soft_labels.sh my_extraction_gpu0 GPU_ID=0
    
    # Run on a Mac (CPU only)
    ./run_docker_soft_labels.sh my_extraction_mac MAC
    ```
    This saves results to `soft_labels_complete/`.

-   **Step 2: Run Distillation Training:**
    ```bash
    # Use GPU 0 for training
    ./run_docker_distillation.sh my_training_gpu0 GPU_ID=0

    # Override parameters
    ./run_docker_distillation.sh my_training_run GPU_ID=1 training.epochs=50 dataset.num_workers=8
    ```
For more details on server usage, GPU management, and troubleshooting, see the comments within the `run_docker_*.sh` scripts. 