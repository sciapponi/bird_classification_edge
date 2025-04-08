# Bird Classification Edge

A comprehensive framework for building and training lightweight neural networks to classify bird sounds, optimized for edge devices.

## Setup Guide

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bird_classification_edge.git
   cd bird_classification_edge
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p bird_sound_dataset logs
   ```

## Data Preparation

### Bird Sound Dataset

To use your bird sound dataset:

1. Create a folder structure like this:
   ```
   bird_sound_dataset/
   ├── species_1/
   │   ├── recording1.wav
   │   ├── recording2.wav
   │   └── ...
   ├── species_2/
   │   ├── recording1.wav
   │   └── ...
   └── ...
   ```

2. Each species should have its own folder containing all relevant audio files.

3. Supported formats: WAV (recommended), MP3, OGG, FLAC.

4. Update the bird classes in your configuration file:
   ```yaml
   dataset:
     allowed_bird_classes: ["species_1", "species_2", ...]
   ```

### ESC-50 Dataset (Non-Bird Sounds)

The ESC-50 dataset is downloaded and extracted automatically when you run training with `download_datasets: true`. This dataset provides 50 classes of environmental sounds, with animal classes filtered out to create the "non-bird" category.

## Bird Call Extraction


1. **Bandpass Filtering**: 
   - Applies a Butterworth filter (default: 2000-10000 Hz range)

2. **Envelope Calculation**:
   - Calculates amplitude envelope using 50ms windows with 10ms overlap

3. **Adaptive Peak Detection**:
   - Uses median and median absolute deviation for robust thresholds

4. **Peak Selection**:
   - Sorts peaks by prominence and keeps those above the threshold

5. **Segment Extraction**:
   - Creates 3-second audio segments centered on detected peaks
