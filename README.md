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
├── generate_no_birds_samples.py  # Script to generate "no_birds" class samples (offline mode)
├── train.py                 # Main training script
├── config/                  # Configuration files
│   └── bird_classification.yaml  # Main configuration
└── [other project files]
```

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

## Usage Examples

### Generate "No Birds" Samples (Offline Mode Step)

```bash
# Basic usage, saves to augmented_dataset/no_birds/
python generate_no_birds_samples.py --bird_dir bird_sound_dataset

# With custom parameters
python generate_no_birds_samples.py \
  --bird_dir bird_sound_dataset \
  --output_dir custom_no_birds_set \
  --num_samples 200 \
  --energy_threshold 2.0 \
  --files_per_class 15
```

### Running Training

- **To use pre-generated "no_birds" samples:**
  1. Ensure `generate_no_birds_samples.py` has been run and samples exist (e.g., in `augmented_dataset/no_birds/`).
  2. In `config/bird_classification.yaml`, set:
     ```yaml
     dataset:
       load_pregenerated_no_birds: true
       pregenerated_no_birds_dir: "augmented_dataset/no_birds/" # or your custom path
       # Optionally adjust num_no_bird_samples if you want to load a specific subset
     ```
  3. Run: `python train.py`

- **To generate "no_birds" samples on-the-fly:**
  1. In `config/bird_classification.yaml`, set:
     ```yaml
     dataset:
       load_pregenerated_no_birds: false
       num_no_bird_samples: 100 # Adjust as needed
       esc50_no_bird_ratio: 0.5   # Adjust as needed
     ```
  2. Run: `python train.py`

### Pre-Training Dataset Inspection

```python
from datasets.empty_segment_dataset import EmptySegmentDataset

# Check how many silent segments can be found with different thresholds
for threshold in [0.5, 1.0, 2.0, 3.0]:
    dataset = EmptySegmentDataset(
        bird_data_dir="bird_sound_dataset",
        allowed_bird_classes=["species1", "species2"],
        no_birds_label=0,
        energy_threshold_factor=threshold
    )
    print(f"Threshold {threshold}: Found {len(dataset)} segments")
```
