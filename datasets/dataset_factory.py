"""
Dataset Factory

This module provides functions for creating complex datasets by combining
different data sources for bird sound classification.
"""

import os
import random
from torch.utils.data import Dataset, ConcatDataset, Subset

class LabelAdapterDataset(Dataset):
    """
    A dataset wrapper that changes the label of all samples in a dataset.
    
    This is useful for adapting datasets to use a different labeling scheme,
    such as treating all ESC-50 samples as a single "no birds" class.
    """
    def __init__(self, subset, label):
        """
        Initialize the label adapter.
        
        Args:
            subset (Dataset): The source dataset
            label (int): The label to assign to all samples
        """
        self.subset = subset
        self.label = label

    def __getitem__(self, index):
        """
        Return the sample at index with the new label.
        
        Args:
            index (int): Index of the sample
            
        Returns:
            tuple: (waveform, new_label)
        """
        x, _ = self.subset[index] # Get data, ignore original label
        return x, self.label      # Return data with the new fixed label

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.subset)

def create_combined_dataset(bird_data_dir, esc50_dir, allowed_bird_classes=None,
                          target_sr=22050, clip_duration=3.0, subset="training",
                          num_no_bird_samples=100,
                          esc50_no_bird_ratio=0.5, 
                          use_augmentation=True,
                          snr_db_range=(5, 15),
                          custom_file_dict=None):
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
        
    Returns:
        ConcatDataset: Combined dataset with bird classes + "no birds" class
    """
    from .bird_dataset import BirdSoundDataset
    from .esc50_dataset import ESC50Dataset
    from .empty_segment_dataset import EmptySegmentDataset
    
    print(f"--- Creating combined dataset: {subset} ---")
    
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
    num_esc50_no_birds = int(num_no_bird_samples * esc50_no_bird_ratio)
    num_empty_segments_no_birds = num_no_bird_samples - num_esc50_no_birds

    print(f"Targeting {num_no_bird_samples} 'no birds' samples: "
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
    if num_esc50_no_birds > 0 and len(datasets_to_combine) > 1:
         print(f"  - ESC-50 'no birds': {len(datasets_to_combine[1])}")
    if num_empty_segments_no_birds > 0 and len(datasets_to_combine) > 2:
         print(f"  - Empty segment 'no birds': {len(datasets_to_combine[2])}")
    elif num_empty_segments_no_birds > 0 and len(datasets_to_combine) > 1 and num_esc50_no_birds == 0:
         print(f"  - Empty segment 'no birds': {len(datasets_to_combine[1])}")

    print(f"--- Combined dataset '{subset}' created successfully with {total_samples} total samples. ---")
    return combined_dataset 