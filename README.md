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
├── generate_no_birds_samples.py  # Script to generate "no_birds" class samples
├── config/                  # Configuration files
│   └── bird_classification.yaml  # Main configuration
└── [other project files]
```

## Data Augmentation Workflow

### 1. Generate "No Birds" Class Samples

The first step is to generate samples for the "no_birds" class:

```bash
python generate_no_birds_samples.py --bird_dir path/to/birds --energy_threshold 2.0 --files_per_class 10
```

This script:
- Creates a balanced set of "no_birds" samples from two sources:
  - **ESC-50 environmental sounds** (rain, footsteps, engines, etc.)
  - **Empty/silent segments** extracted from bird recordings
- Saves these to `augmented_dataset/no_birds/`

### 2. Dataset Structure After Generation

```
bird_sound_dataset/         # Original dataset
  ├── species1/             # Bird class folders
  │   ├── recording1.wav
  │   └── ...
  └── species2/
      └── ...

augmented_dataset/          # Generated data
  └── no_birds/             # The "no birds" class
      ├── esc50_0000.wav    # Environmental sounds
      ├── ...
      ├── empty_0000.wav    # Silent segments
      └── ...
```

### 3. Dataset Loading for Training

During model training, the system:
1. Loads bird sounds from each species folder
2. Loads the "no_birds" samples
3. Applies augmentations during training:
   - Bird call extraction
   - Background mixing with controlled SNR
   - Time/frequency masking
   - Speed perturbation
   - Time shifting

## Key Components

### EmptySegmentDataset

A dataset class that finds and extracts segments with low energy (likely no bird calls) from audio recordings:

```python
empty_dataset = EmptySegmentDataset(
    bird_data_dir="bird_recordings/",
    allowed_bird_classes=["species1", "species2"],
    no_birds_label=0,  # Label index for "no birds" class
    energy_threshold_factor=1.5  # Higher values detect more segments
)
```

### BirdSoundDataset

Handles loading, processing, and augmentation of bird sound recordings:

```python
bird_dataset = BirdSoundDataset(
    root_dir="bird_recordings/",
    allowed_classes=["species1", "species2"],
    extract_calls=True,  # Automatically find bird calls in recordings
    augment=True  # Apply augmentations during training
)
```

## Script Parameters

### generate_no_birds_samples.py

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

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download bird recordings or use your own dataset
4. Generate the "no_birds" class samples
5. Configure training parameters in `config/bird_classification.yaml`
6. Run training (see training documentation)

## Usage Examples

### Generate "No Birds" Samples

```bash
# Basic usage
python generate_no_birds_samples.py --bird_dir bird_sound_dataset

# With custom parameters
python generate_no_birds_samples.py \
  --bird_dir bird_sound_dataset \
  --output_dir custom_dataset \
  --num_samples 200 \
  --energy_threshold 2.0 \
  --files_per_class 15
```

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
