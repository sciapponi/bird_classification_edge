"""
Dataset Factory

This module provides functions for creating complex datasets by combining
different data sources for bird sound classification.
"""

import os
import random
import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchaudio # Import torchaudio
from pathlib import Path # Import Path

# Definition for LabelAdapterDataset
class LabelAdapterDataset(Dataset):
    def __init__(self, wrapped_dataset, new_label):
        self.wrapped_dataset = wrapped_dataset
        self.new_label = new_label

    def __len__(self):
        return len(self.wrapped_dataset)

    def __getitem__(self, idx):
        data, _ = self.wrapped_dataset[idx] # Original label is ignored
        return data, self.new_label

# Nuova classe Dataset per caricare file .wav pre-generati
class PreGeneratedNoBirdsDataset(Dataset):
    """
    Dataset for loading pre-generated 'no birds' audio files from a directory.
    """
    def __init__(self, root_dir, no_birds_label, target_sr=22050, clip_duration=3.0):
        self.root_dir = Path(root_dir)
        self.no_birds_label = no_birds_label
        self.target_sr = target_sr
        self.target_len_samples = int(target_sr * clip_duration)
        
        self.audio_files = []
        if self.root_dir.exists() and self.root_dir.is_dir():
            self.audio_files = sorted([f for f in self.root_dir.iterdir() if f.suffix.lower() in ['.wav', '.mp3', '.ogg', '.flac']])
        
        if not self.audio_files:
            print(f"WARNING: No audio files found in PreGeneratedNoBirdsDataset directory: {root_dir}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a silent tensor of the correct shape as a fallback
            return torch.zeros((1, self.target_len_samples)), self.no_birds_label

        # Resample if necessary
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Adjust length (pad or truncate)
        current_len_samples = waveform.shape[1]
        if current_len_samples > self.target_len_samples:
            waveform = waveform[:, :self.target_len_samples]
        elif current_len_samples < self.target_len_samples:
            padding = self.target_len_samples - current_len_samples
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform, self.no_birds_label

def create_no_birds_dataset(num_samples, 
                              no_birds_label, 
                              esc50_dir, 
                              bird_data_dir, # Needed for empty segments
                              allowed_bird_classes, # Needed for empty segments
                              subset, # Needed for ESC-50 folds and empty segment source logic (if refined)
                              target_sr, 
                              clip_duration,
                              esc50_no_bird_ratio, 
                              load_pregenerated, 
                              pregenerated_dir):
    """
    Creates a dataset containing only "no birds" samples, either by loading 
    pre-generated files or generating them on-the-fly from ESC-50 and empty segments.
    
    Args:
        num_samples (int): Target number of "no birds" samples.
        no_birds_label (int): Label to assign to these samples.
        esc50_dir (str): Path to ESC-50 dataset.
        bird_data_dir (str): Path to bird sound dataset (for empty segments).
        allowed_bird_classes (list): List of bird classes (for empty segments source).
        subset (str): 'training', 'validation', or 'testing' (for ESC-50 folds).
        target_sr (int): Target sample rate.
        clip_duration (float): Target clip duration.
        esc50_no_bird_ratio (float): Proportion of samples from ESC-50.
        load_pregenerated (bool): Whether to load from pregenerated_dir.
        pregenerated_dir (str): Directory containing pre-generated samples.
        
    Returns:
        Dataset: A PyTorch Dataset containing the "no birds" samples.
                 Returns None if no samples could be created.
    """
    from .esc50_dataset import ESC50Dataset
    from .empty_segment_dataset import EmptySegmentDataset
    
    datasets_to_add = []
    actual_total_no_birds = 0

    print(f"--- Creating 'no birds' dataset for subset: {subset} ---")
    print(f"Target samples: {num_samples}")
    print(f"Mode: {'Loading pre-generated' if load_pregenerated else 'Generating on-the-fly'}")

    if load_pregenerated:
        print(f"Loading pre-generated 'no birds' samples from {pregenerated_dir}...")
        pregen_dataset = PreGeneratedNoBirdsDataset(
            root_dir=pregenerated_dir,
            no_birds_label=no_birds_label,
            target_sr=target_sr,
            clip_duration=clip_duration
        )
        
        if len(pregen_dataset) == 0:
            print(f"  WARNING: No pre-generated 'no birds' files found in {pregenerated_dir}.")
            return None # Return None if no pregenerated samples found
        
        # FIXED: Apply train/val/test split to pregenerated samples
        # Use consistent split ratios (70/15/15) and seed for reproducibility
        total_pregen = len(pregen_dataset)
        split_seed = 42  # Use consistent seed for reproducible splits
        
        # Calculate split sizes
        import math
        val_split = 0.15
        test_split = 0.15
        
        num_val = math.floor(val_split * total_pregen)
        num_test = math.floor(test_split * total_pregen)  
        num_train = total_pregen - num_val - num_test
        
        print(f"  Total pre-generated samples: {total_pregen}")
        print(f"  Split: Train={num_train}, Val={num_val}, Test={num_test}")
        
        # Create deterministic split using seed
        rng = random.Random(split_seed)
        all_indices = list(range(total_pregen))
        rng.shuffle(all_indices)
        
        # Split indices
        if subset == "training":
            subset_indices = all_indices[:num_train]
        elif subset == "validation":
            subset_indices = all_indices[num_train:num_train + num_val]
        elif subset == "testing":
            subset_indices = all_indices[num_train + num_val:]
        else:
            raise ValueError(f"Invalid subset: {subset}")
        
        print(f"  Using {len(subset_indices)} samples for subset '{subset}'")
        
        # Apply num_samples limit if specified
        if num_samples > 0 and len(subset_indices) > num_samples:
            print(f"  Limiting to {num_samples} samples as requested")
            subset_indices = subset_indices[:num_samples]
        
        if len(subset_indices) > 0:
            final_pregen_dataset = Subset(pregen_dataset, subset_indices)
            datasets_to_add.append(final_pregen_dataset)
            actual_total_no_birds = len(final_pregen_dataset)
            print(f"  Added {actual_total_no_birds} pre-generated 'no birds' samples for {subset}.")
        else:
            print(f"  No pre-generated samples available for subset '{subset}'.")
            return None

    else: # Generate on-the-fly
        num_esc50_target = int(num_samples * esc50_no_bird_ratio)
        num_empty_target = num_samples - num_esc50_target

        print(f"Generating on-the-fly: {num_esc50_target} from ESC-50, {num_empty_target} from empty segments.")

        # Define ESC-50 categories and folds
        NO_BIRD_CATEGORIES = [
            'rain', 'sea_waves', 'crackling_fire', 'crickets', 'water_drops', 'wind',
            'footsteps', 'door_wood_creaks', 'car_horn', 'engine', 'train', 
            'dog', 'cat', 'frog'
        ]
        if subset == "training": folds = [1, 2, 3]
        elif subset == "validation": folds = [4]
        else: folds = [5] # testing

        # 1. Get from ESC-50
        if num_esc50_target > 0:
            esc50_base = ESC50Dataset(
                root_dir=esc50_dir, fold=folds,
                select_categories=NO_BIRD_CATEGORIES,
                target_sr=target_sr, target_length=clip_duration
            )
            if len(esc50_base) < num_esc50_target:
                print(f"  WARNING: Requested {num_esc50_target} ESC-50, only {len(esc50_base)} available. Using all.")
                num_esc50_actual = len(esc50_base)
            else:
                num_esc50_actual = num_esc50_target
            
            if num_esc50_actual > 0:
                 esc50_indices = random.sample(range(len(esc50_base)), num_esc50_actual)
                 esc50_subset = Subset(esc50_base, esc50_indices) 
                 esc50_no_birds_dataset = LabelAdapterDataset(esc50_subset, no_birds_label)
                 datasets_to_add.append(esc50_no_birds_dataset)
                 actual_total_no_birds += len(esc50_no_birds_dataset)
                 print(f"  Added {len(esc50_no_birds_dataset)} 'no birds' from ESC-50.")
            else:
                 print("  Skipping ESC-50 'no birds' samples.")

        # 2. Get from empty segments
        if num_empty_target > 0:
            # TODO: Refine empty segment source based on subset? 
            # Currently uses all bird_data_dir files. Needs BirdSoundDataset split logic first.
            # For now, we generate from the *entire* bird_data_dir for simplicity, accepting potential minor leakage here.
            # A better approach would be to pass the specific file list for the *bird* subset here.
            # print(f"  Generating {num_empty_target} empty segments (from all bird files - potential minor leakage for val/test)...")
            # The EmptySegmentDataset now handles its own printing regarding master pool etc.
            
            # Get the global seed from hydra config if possible, or use a default
            # This assumes hydra config is accessible or a fixed seed is okay here.
            # For now, let's use a fixed seed or pass one in. We'll use 42 for consistency for now.
            # In a more integrated setup, this seed would come from the main config.
            current_seed = 42 # Placeholder, ideally from main config.cfg.training.seed
            
            empty_segment_base = EmptySegmentDataset(
                bird_data_dir=bird_data_dir,
                allowed_bird_classes=allowed_bird_classes,
                no_birds_label=no_birds_label,
                num_target_segments=num_empty_target, # Pass the target for this instance
                clip_duration=clip_duration,
                sr=target_sr,
                max_segments_per_file=5, # This is max per file during initial scan
                energy_threshold_factor=0.5, # Default, can be configured
                custom_audio_files=None, # For now, master pool scans all bird_data_dir
                seed=current_seed
            )
            if len(empty_segment_base) > 0:
                # No need to sample further with Subset, EmptySegmentDataset instance already has its target segments
                datasets_to_add.append(empty_segment_base)
                actual_total_no_birds += len(empty_segment_base)
                print(f"  Added {len(empty_segment_base)} 'no birds' from empty segments pool.")
            else:
                print(f"  WARNING: EmptySegmentDataset provided 0 segments for the target of {num_empty_target}.")

    # Combine the created "no birds" datasets (if any)
    if not datasets_to_add:
        print(f"--- No 'no birds' samples created for subset: {subset} --- ")
        return None
    elif len(datasets_to_add) == 1:
        final_dataset = datasets_to_add[0]
    else:
        final_dataset = ConcatDataset(datasets_to_add)
        
    print(f"--- 'No birds' dataset created for subset: {subset} with {actual_total_no_birds} samples --- ")
    return final_dataset

def create_combined_dataset(bird_data_dir, esc50_dir, allowed_bird_classes=None,
                          target_sr=22050, clip_duration=3.0, subset="training",
                          num_no_bird_samples=100,
                          esc50_no_bird_ratio=0.5, 
                          use_augmentation=True,
                          snr_db_range=(5, 15),
                          custom_file_dict=None,
                          load_pregenerated_no_birds: bool = False, # Nuovo parametro
                          pregenerated_no_birds_dir: str = "augmented_dataset/no_birds/", # Nuovo parametro
                          validation_split=0.15,
                          test_split=0.15,
                          split_seed=42
                          ):
    """
    Create a combined dataset with bird sounds and "no birds" class.
    
    The dataset includes:
    1. Bird sound segments with optional augmentation
    2. ESC-50 environmental sounds as "no birds" examples
    3. Silent segments from bird recordings as additional "no birds" examples
    
    Args:
        bird_data_dir (str): Directory containing bird sound data
        esc50_dir (str): Directory containing ESC-50 dataset
        allowed_bird_classes (list): List of bird classes to include (None for all)
        target_sr (int): Target sample rate for all audio
        clip_duration (float): Target duration for audio clips in seconds
        subset (str): Dataset subset ('training', 'validation', 'testing')
        num_no_bird_samples (int): Total number of "no birds" samples to include
        esc50_no_bird_ratio (float): Proportion of "no birds" samples from ESC-50 (0.0-1.0)
        use_augmentation (bool): Whether to apply augmentation to bird sounds
        snr_db_range (tuple): Min and Max SNR (dB) for background mixing
        custom_file_dict (dict): Dictionary mapping class names to file paths
        load_pregenerated_no_birds (bool): Whether to load pre-generated "no birds" samples
        pregenerated_no_birds_dir (str): Directory containing pre-generated "no birds" samples
        validation_split (float): Proportion of training data to use for validation
        test_split (float): Proportion of training data to use for testing
        split_seed (int): Seed for random splitting
        
    Returns:
        ConcatDataset: Combined dataset with bird classes + "no birds" class
    """
    from .bird_dataset import BirdSoundDataset
    from .esc50_dataset import ESC50Dataset
    from .empty_segment_dataset import EmptySegmentDataset
    
    print(f"--- Creating combined dataset: {subset} ---")
    print(f"Mode for 'no birds' samples: {'Loading pre-generated' if load_pregenerated_no_birds else 'Generating on-the-fly'}")
    if load_pregenerated_no_birds:
        print(f"Pre-generated 'no birds' directory: {pregenerated_no_birds_dir}")
    
    # Define transform (can be expanded later if needed)
    transform = None

    # Determine bird classes if not specified
    if not allowed_bird_classes:
        print(f"No specific bird classes provided, scanning {bird_data_dir}...")
        allowed_bird_classes = [d for d in os.listdir(bird_data_dir)
                              if os.path.isdir(os.path.join(bird_data_dir, d))]
        print(f"Found bird classes: {allowed_bird_classes}")
        
    num_bird_classes = len(allowed_bird_classes)
    NO_BIRDS_LABEL = num_bird_classes # Label for the "no birds" class
    print(f"Number of bird classes: {num_bird_classes}, 'No Birds' Label: {NO_BIRDS_LABEL}")

    # Define ESC-50 categories for "no birds" class and background noise
    NO_BIRD_CATEGORIES = [
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'water_drops', 'wind',
        'footsteps', 'door_wood_creaks', 'car_horn', 'engine', 'train', 
        'dog', 'cat', 'frog'
    ]
    print(f"Using ESC-50 categories for 'no birds'/background: {NO_BIRD_CATEGORIES}")

    # Determine ESC-50 folds based on subset
    if subset == "training":
        folds = [1, 2, 3]
    elif subset == "validation":
        folds = [4]
    else:  # testing
        folds = [5]
    print(f"Using ESC-50 folds: {folds} for subset '{subset}'")

    # Create the base ESC-50 dataset filtered for the required categories
    esc50_background_dataset = ESC50Dataset(
        root_dir=esc50_dir,
        fold=folds,
        select_categories=NO_BIRD_CATEGORIES,
        transform=transform,
        target_sr=target_sr,
        target_length=clip_duration
    )
    print(f"Created base ESC-50 dataset for background/no-birds with {len(esc50_background_dataset)} samples.")

    # --- Create Bird Sound Dataset ---
    # Augmentation is only applied during training and requires the background dataset
    apply_bird_augmentation = use_augmentation and subset == "training"
    bird_dataset = BirdSoundDataset(
        root_dir=bird_data_dir,
        transform=transform,
        allowed_classes=allowed_bird_classes,
        subset=subset,
        validation_split=validation_split,
        test_split=test_split,
        split_seed=split_seed,
        augment=apply_bird_augmentation,
        sr=target_sr,
        clip_duration=clip_duration,
        extract_calls=True, # Use call extraction for bird sounds
        # Pass the background dataset for mixing augmentation
        background_dataset=esc50_background_dataset if apply_bird_augmentation else None,
        custom_file_dict=custom_file_dict
    )
    print(f"Created bird sound dataset with {len(bird_dataset)} samples. Augmentation: {apply_bird_augmentation}")

        # --- Create "No Birds" Samples ---
    datasets_to_combine = [bird_dataset]
    
    # Use the fixed create_no_birds_dataset function that properly handles splits
    no_birds_dataset = create_no_birds_dataset(
        num_samples=num_no_bird_samples,
        no_birds_label=NO_BIRDS_LABEL,
        esc50_dir=esc50_dir,
        bird_data_dir=bird_data_dir,
        allowed_bird_classes=allowed_bird_classes,
        subset=subset,
        target_sr=target_sr,
        clip_duration=clip_duration,
        esc50_no_bird_ratio=esc50_no_bird_ratio,
        load_pregenerated=load_pregenerated_no_birds,
        pregenerated_dir=pregenerated_no_birds_dir
    )
    
    if no_birds_dataset is not None:
        datasets_to_combine.append(no_birds_dataset)
        print(f"  Added {len(no_birds_dataset)} 'no birds' samples to combined dataset.")
    else:
        print(f"  No 'no birds' samples added to combined dataset.")

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets_to_combine)

    # Print dataset statistics
    total_samples = len(combined_dataset)
    print(f"  - Bird samples: {len(bird_dataset)}")
    
    # Calculate 'no birds' sample count
    no_birds_sample_count = 0
    if len(datasets_to_combine) > 1:
        no_birds_sample_count = len(datasets_to_combine[1])  # The no_birds_dataset
        mode_text = "Pre-generated" if load_pregenerated_no_birds else "On-the-fly"
        print(f"  - {mode_text} 'no birds': {no_birds_sample_count}")
            
    print(f"--- Combined dataset '{subset}' created successfully with {total_samples} total samples. ({no_birds_sample_count} 'no birds' samples) ---")
    return combined_dataset 