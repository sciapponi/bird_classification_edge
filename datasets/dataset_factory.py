"""
Dataset Factory

This module provides functions for creating complex datasets by combining
different data sources for bird sound classification.
"""

import os
import random
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
        
        # Select a subset if num_samples is restrictive
        if num_samples > 0:
            if len(pregen_dataset) < num_samples:
                print(f"  WARNING: Requested {num_samples} pre-generated, but only {len(pregen_dataset)} available. Using all.")
                actual_num_samples = len(pregen_dataset)
            else:
                actual_num_samples = num_samples
            
            if actual_num_samples > 0:
                pregen_indices = random.sample(range(len(pregen_dataset)), actual_num_samples)
                final_pregen_dataset = Subset(pregen_dataset, pregen_indices)
                datasets_to_add.append(final_pregen_dataset)
                actual_total_no_birds = len(final_pregen_dataset)
                print(f"  Added {actual_total_no_birds} pre-generated 'no birds' samples.")
            else:
                 print("  No pre-generated samples to add (num_samples=0).")
                 return None # Return None if 0 samples requested
        else: # Use all available pre-generated samples
             datasets_to_add.append(pregen_dataset)
             actual_total_no_birds = len(pregen_dataset)
             print(f"  Added all {actual_total_no_birds} available pre-generated 'no birds' samples.")

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
    
    if load_pregenerated_no_birds:
        print(f"Loading pre-generated 'no birds' samples from {pregenerated_no_birds_dir}...")
        pregen_no_birds_dataset = PreGeneratedNoBirdsDataset(
            root_dir=pregenerated_no_birds_dir,
            no_birds_label=NO_BIRDS_LABEL,
            target_sr=target_sr,
            clip_duration=clip_duration
        )
        
        if len(pregen_no_birds_dataset) == 0:
            print(f"WARNING: No pre-generated 'no birds' files found in {pregenerated_no_birds_dir}. "
                  f"Consider running generate_no_birds_samples.py or setting load_pregenerated_no_birds=False.")
        
        # If num_no_bird_samples is specified, select a subset
        # Otherwise, use all available pre-generated samples
        if num_no_bird_samples > 0 and len(pregen_no_birds_dataset) > 0:
            if len(pregen_no_birds_dataset) < num_no_bird_samples:
                print(f"  WARNING: Requested {num_no_bird_samples} pre-generated 'no birds' samples, "
                      f"but only {len(pregen_no_birds_dataset)} available. Using all available.")
                actual_num_samples = len(pregen_no_birds_dataset)
            else:
                actual_num_samples = num_no_bird_samples
            
            if actual_num_samples > 0:
                pregen_indices = random.sample(range(len(pregen_no_birds_dataset)), actual_num_samples)
                final_pregen_no_birds_dataset = Subset(pregen_no_birds_dataset, pregen_indices)
                datasets_to_combine.append(final_pregen_no_birds_dataset)
                print(f"  Added {len(final_pregen_no_birds_dataset)} pre-generated 'no birds' samples.")
            else:
                print(f"  No pre-generated 'no birds' samples to add (either none found or num_samples=0).")
        elif len(pregen_no_birds_dataset) > 0: # num_no_bird_samples is 0 or not restrictive, use all
            datasets_to_combine.append(pregen_no_birds_dataset)
            print(f"  Added all {len(pregen_no_birds_dataset)} available pre-generated 'no birds' samples.")
        else:
            print(f"  No pre-generated 'no birds' samples to add.")

    else: # Generate 'no birds' samples on-the-fly
        num_esc50_no_birds = int(num_no_bird_samples * esc50_no_bird_ratio)
        num_empty_segments_no_birds = num_no_bird_samples - num_esc50_no_birds

        print(f"Targeting {num_no_bird_samples} 'no birds' samples (on-the-fly): "
              f"{num_esc50_no_birds} from ESC-50, {num_empty_segments_no_birds} from empty segments.")

        # 1. Get "no birds" samples from ESC-50
        if num_esc50_no_birds > 0:
            if len(esc50_background_dataset) < num_esc50_no_birds:
                print(f"WARNING: Requested {num_esc50_no_birds} ESC-50 'no birds' samples, "
                      f"but only {len(esc50_background_dataset)} available. Using all available.")
                num_esc50_no_birds = len(esc50_background_dataset)

            if num_esc50_no_birds > 0:
                 # Randomly select indices from the base ESC-50 dataset
                 esc50_indices = random.sample(range(len(esc50_background_dataset)), num_esc50_no_birds)
                 # Create a subset using torch.utils.data.Subset
                 esc50_subset = Subset(esc50_background_dataset, esc50_indices) 
                 # Wrap subset to assign the correct "no birds" label
                 esc50_no_birds_dataset = LabelAdapterDataset(esc50_subset, NO_BIRDS_LABEL)
                 datasets_to_combine.append(esc50_no_birds_dataset)
                 print(f"Added {len(esc50_no_birds_dataset)} 'no birds' samples from ESC-50.")
            else:
                 print("Skipping ESC-50 'no birds' samples as none are available or requested.")

        # 2. Get "no birds" samples from empty segments
        if num_empty_segments_no_birds > 0:
            print("Creating dataset of empty segments from bird recordings...")
            
            # If we have a custom_file_dict, use only those files for finding empty segments
            if custom_file_dict is not None:
                print("Using custom file list for finding empty segments")
                # Flatten the file list for EmptySegmentDataset
                all_audio_files = []
                for class_files in custom_file_dict.values():
                    all_audio_files.extend(class_files)
                
                # Create a custom empty segment dataset that processes specific files
                empty_segment_full_dataset = EmptySegmentDataset(
                    bird_data_dir=bird_data_dir,
                    allowed_bird_classes=allowed_bird_classes,
                    no_birds_label=NO_BIRDS_LABEL,
                    clip_duration=clip_duration,
                    sr=target_sr,
                    lowcut=2000,
                    highcut=10000,
                    max_segments_per_file=5,
                    custom_audio_files=all_audio_files
                )
            else:
                # Standard empty segment extraction from all files
                empty_segment_full_dataset = EmptySegmentDataset(
                    bird_data_dir=bird_data_dir,
                    allowed_bird_classes=allowed_bird_classes,
                    no_birds_label=NO_BIRDS_LABEL,
                    clip_duration=clip_duration,
                    sr=target_sr,
                    lowcut=2000,
                    highcut=10000,
                    max_segments_per_file=5
                )
            
            # Check if we found enough empty segments
            if len(empty_segment_full_dataset) < num_empty_segments_no_birds:
                print(f"WARNING: Requested {num_empty_segments_no_birds} empty segments, "
                      f"but only found {len(empty_segment_full_dataset)}. Using all available.")
                num_empty_segments_no_birds = len(empty_segment_full_dataset)

            if num_empty_segments_no_birds > 0:
                empty_indices = random.sample(range(len(empty_segment_full_dataset)), num_empty_segments_no_birds)
                empty_subset = Subset(empty_segment_full_dataset, empty_indices)
                datasets_to_combine.append(empty_subset)
                print(f"Added {len(empty_subset)} 'no birds' samples from empty segments.")
            else:
                print("Skipping empty segment 'no birds' samples as none are available or requested.")

    # Combine all datasets
    combined_dataset = ConcatDataset(datasets_to_combine)

    # Print dataset statistics
    total_samples = len(combined_dataset)
    print(f"  - Bird samples: {len(bird_dataset)}")
    
    # Adjust printing of 'no birds' samples based on the mode
    no_birds_sample_count = 0
    if load_pregenerated_no_birds:
        if len(datasets_to_combine) > 1 and isinstance(datasets_to_combine[-1], (Subset, PreGeneratedNoBirdsDataset)):
            no_birds_sample_count = len(datasets_to_combine[-1])
            print(f"  - Pre-generated 'no birds': {no_birds_sample_count}")
    else:
        # This part needs to be careful about indices if only one type of on-the-fly no_birds is generated
        idx_offset = 1
        if num_esc50_no_birds > 0 and len(datasets_to_combine) > idx_offset:
            actual_esc50_count = len(datasets_to_combine[idx_offset])
            print(f"  - ESC-50 'no birds' (on-the-fly): {actual_esc50_count}")
            no_birds_sample_count += actual_esc50_count
            idx_offset +=1
        if num_empty_segments_no_birds > 0 and len(datasets_to_combine) > idx_offset:
            actual_empty_count = len(datasets_to_combine[idx_offset])
            print(f"  - Empty segment 'no birds' (on-the-fly): {actual_empty_count}")
            no_birds_sample_count += actual_empty_count
            
    print(f"--- Combined dataset '{subset}' created successfully with {total_samples} total samples. ({no_birds_sample_count} 'no birds' samples) ---")
    return combined_dataset 