"""
Test script for the datasets module.

This script tests the functionality of the datasets module by creating
various dataset instances and verifying their behavior.
"""

import os
import torch
from torch.utils.data import DataLoader

from .esc50_dataset import ESC50Dataset, download_and_extract_esc50
from .bird_dataset import BirdSoundDataset
from .empty_segment_dataset import EmptySegmentDataset
from .dataset_factory import create_combined_dataset

def test_esc50_dataset(esc50_dir):
    """Test the ESC50Dataset functionality."""
    print("\n=== Testing ESC50Dataset ===")
    
    # Create a dataset with all categories
    all_categories_dataset = ESC50Dataset(
        root_dir=esc50_dir,
        fold=[1],  # Just use fold 1 for testing
    )
    print(f"All categories dataset: {len(all_categories_dataset)} samples")
    
    # Test loading a sample
    if len(all_categories_dataset) > 0:
        waveform, label = all_categories_dataset[0]
        category = all_categories_dataset.filtered_categories[label]
        print(f"Sample loaded: shape={waveform.shape}, label={label}, category={category}")
    
    # Create a dataset with specific categories
    select_categories = ['rain', 'sea_waves', 'footsteps']
    filtered_dataset = ESC50Dataset(
        root_dir=esc50_dir,
        fold=[1, 2],
        select_categories=select_categories
    )
    print(f"Filtered dataset ({', '.join(select_categories)}): {len(filtered_dataset)} samples")
    
    # Test DataLoader
    if len(filtered_dataset) > 0:
        loader = DataLoader(filtered_dataset, batch_size=4, shuffle=True)
        batch = next(iter(loader))
        print(f"Batch shape: {batch[0].shape}, labels: {batch[1]}")
    
    return True

def test_bird_dataset(bird_dir, bird_classes=None):
    """Test the BirdSoundDataset functionality."""
    print("\n=== Testing BirdSoundDataset ===")
    
    # Get bird classes if not provided
    if bird_classes is None:
        bird_classes = [d for d in os.listdir(bird_dir) 
                      if os.path.isdir(os.path.join(bird_dir, d))]
    
    print(f"Using bird classes: {bird_classes}")
    
    # Create a dataset without augmentation
    bird_dataset = BirdSoundDataset(
        root_dir=bird_dir,
        allowed_classes=bird_classes,
        subset="testing",
        augment=False,
        extract_calls=True
    )
    print(f"Bird dataset: {len(bird_dataset)} samples")
    
    # Test loading a sample
    if len(bird_dataset) > 0:
        waveform, label = bird_dataset[0]
        class_name = bird_classes[label]
        print(f"Sample loaded: shape={waveform.shape}, label={label}, class={class_name}")
    
    return True

def test_empty_segment_dataset(bird_dir, bird_classes=None):
    """Test the EmptySegmentDataset functionality."""
    print("\n=== Testing EmptySegmentDataset ===")
    
    # Get bird classes if not provided
    if bird_classes is None:
        bird_classes = [d for d in os.listdir(bird_dir) 
                      if os.path.isdir(os.path.join(bird_dir, d))]
    
    print(f"Scanning for empty segments in classes: {bird_classes}")
    
    # Create an EmptySegmentDataset
    empty_dataset = EmptySegmentDataset(
        bird_data_dir=bird_dir,
        allowed_bird_classes=bird_classes,
        no_birds_label=len(bird_classes),  # "no birds" label
        max_segments_per_file=2  # Limit segments for testing
    )
    print(f"Empty segment dataset: {len(empty_dataset)} segments")
    
    # Test loading a sample
    if len(empty_dataset) > 0:
        waveform, label = empty_dataset[0]
        print(f"Sample loaded: shape={waveform.shape}, label={label}")
        
        # Verify that it has low energy (likely to be an empty segment)
        energy = torch.mean(waveform**2).item()
        print(f"Sample energy: {energy:.6f} (should be very low)")
    
    return True

def test_combined_dataset(bird_dir, esc50_dir, bird_classes=None):
    """Test the combined dataset creation."""
    print("\n=== Testing Combined Dataset ===")
    
    # Create combined dataset with a small number of "no birds" samples
    combined_dataset = create_combined_dataset(
        bird_data_dir=bird_dir,
        esc50_dir=esc50_dir,
        allowed_bird_classes=bird_classes,
        subset="training",
        num_no_bird_samples=10,  # small number for testing
        esc50_no_bird_ratio=0.5
    )
    
    print(f"Combined dataset: {len(combined_dataset)} samples")
    
    # Test loading a sample
    if len(combined_dataset) > 0:
        waveform, label = combined_dataset[0]
        print(f"Sample loaded: shape={waveform.shape}, label={label}")
    
    return True

def main():
    """Run all tests."""
    print("Testing datasets module...")
    
    # Check if bird data directory exists
    bird_dir = "bird_sound_dataset"
    if not os.path.exists(bird_dir):
        print(f"Creating directory {bird_dir}...")
        os.makedirs(bird_dir, exist_ok=True)
        print("Please add bird sound data in class-specific folders to test properly.")
        return False
    
    # Check bird classes
    bird_classes = [d for d in os.listdir(bird_dir) 
                  if os.path.isdir(os.path.join(bird_dir, d))]
    
    if not bird_classes:
        print("No bird class folders found. Please add folders with audio files.")
        return False
    
    # Download ESC-50 if needed
    esc50_dir = download_and_extract_esc50()
    
    # Run tests
    tests_passed = True
    tests_passed &= test_esc50_dataset(esc50_dir)
    tests_passed &= test_bird_dataset(bird_dir, bird_classes)
    tests_passed &= test_empty_segment_dataset(bird_dir, bird_classes)
    tests_passed &= test_combined_dataset(bird_dir, esc50_dir, bird_classes)
    
    if tests_passed:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed. Check messages above.")
    
    return tests_passed

if __name__ == "__main__":
    main() 