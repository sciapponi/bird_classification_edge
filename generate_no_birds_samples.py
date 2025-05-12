#!/usr/bin/env python
"""
Generate No Birds Samples

This script generates audio samples for the "no birds" class by:
1. Extracting empty segments from bird sound recordings
2. Using selected environmental sounds from the ESC-50 dataset

These samples are saved to disk for later use in training a classifier.
"""

import os
import argparse
import random
import torch
import torchaudio
from pathlib import Path

from datasets.esc50_dataset import ESC50Dataset, download_and_extract_esc50
from datasets.empty_segment_dataset import EmptySegmentDataset

def save_no_birds_samples(bird_data_dir, esc50_dir, output_dir, 
                          num_samples=100, esc50_ratio=0.5,
                          files_per_bird_class=None, allowed_bird_classes=None,
                          energy_threshold_factor=1.5):
    """
    Generate and save audio samples for the "no birds" class.
    
    Args:
        bird_data_dir: Directory containing bird sound recordings
        esc50_dir: Directory containing ESC-50 dataset
        output_dir: Directory to save the generated samples
        num_samples: Number of "no birds" samples to generate
        esc50_ratio: Proportion of samples to source from ESC-50 (vs. empty segments)
        files_per_bird_class: Maximum number of files to scan per bird class
        allowed_bird_classes: List of bird classes to scan (None for all)
        energy_threshold_factor: Factor to multiply median energy for silence threshold (higher values make it easier to find silence)
    """
    print(f"Generating {num_samples} 'no birds' samples:")
    
    # Parameters
    target_sr = 22050
    clip_duration = 3.0
    
    # Define ESC-50 categories to use for "no birds" class
    no_bird_categories = [
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'water_drops', 'wind',
        'footsteps', 'door_wood_creaks', 'car_horn', 'engine', 'train', 
        'dog', 'cat', 'frog'
    ]
    
    # Determine bird classes if not specified
    if not allowed_bird_classes:
        allowed_bird_classes = [d for d in os.listdir(bird_data_dir) 
                              if os.path.isdir(os.path.join(bird_data_dir, d))]
        print(f"Found bird classes: {allowed_bird_classes}")
    
    # Calculate number of samples from each source
    num_esc50_samples = int(num_samples * esc50_ratio)
    num_empty_samples = num_samples - num_esc50_samples
    
    print(f"- {num_esc50_samples} samples from ESC-50 environmental sounds")
    print(f"- {num_empty_samples} samples from empty segments in bird recordings")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Get ESC-50 samples ---
    if num_esc50_samples > 0:
        print("\nCollecting ESC-50 environmental sound samples...")
        
        # Create ESC-50 dataset with the selected categories
        esc50_dataset = ESC50Dataset(
            root_dir=esc50_dir,
            fold=[1, 2, 3],  # Use training folds
            select_categories=no_bird_categories,
            target_sr=target_sr,
            target_length=clip_duration
        )
        
        # Check if we have enough samples
        if len(esc50_dataset) < num_esc50_samples:
            print(f"WARNING: Requested {num_esc50_samples} ESC-50 samples, but only {len(esc50_dataset)} available.")
            num_esc50_samples = len(esc50_dataset)
        
        # Randomly select indices
        esc50_indices = random.sample(range(len(esc50_dataset)), num_esc50_samples)
        
        # Save the selected samples
        for i, idx in enumerate(esc50_indices):
            waveform, _ = esc50_dataset[idx]
            
            # Ensure 2D tensor
            if waveform.ndim > 2:
                waveform = waveform.squeeze(0)
            
            # Save as WAV file
            output_path = os.path.join(output_dir, f"esc50_{i:04d}.wav")
            torchaudio.save(
                output_path,
                waveform,
                sample_rate=target_sr
            )
            
            # Print progress
            if (i + 1) % 10 == 0 or i + 1 == num_esc50_samples:
                print(f"  Saved {i + 1}/{num_esc50_samples} ESC-50 samples")
    
    # --- Get empty segments from bird recordings ---
    if num_empty_samples > 0:
        print("\nExtracting empty segments from bird recordings...")
        print(f"Using energy threshold factor: {energy_threshold_factor}")
        
        # If we need to limit files per bird class
        custom_audio_files = None
        if files_per_bird_class is not None:
            print(f"Limiting to {files_per_bird_class} files per bird class")
            custom_audio_files = []
            for bird_class in allowed_bird_classes:
                class_dir = os.path.join(bird_data_dir, bird_class)
                if os.path.isdir(class_dir):
                    audio_files = [f for f in os.listdir(class_dir) 
                                  if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
                    
                    if len(audio_files) > files_per_bird_class:
                        audio_files = random.sample(audio_files, files_per_bird_class)
                    
                    custom_audio_files.extend([os.path.join(class_dir, f) for f in audio_files])
                    print(f"  Added {len(audio_files)} files from {bird_class}")
        
        # Create empty segment dataset
        # Use a dummy label (0) since we're only interested in the audio content
        empty_dataset = EmptySegmentDataset(
            bird_data_dir=bird_data_dir,
            allowed_bird_classes=allowed_bird_classes,
            no_birds_label=0,  # dummy label
            clip_duration=clip_duration,
            sr=target_sr,
            lowcut=2000,
            highcut=10000,
            max_segments_per_file=5,
            energy_threshold_factor=energy_threshold_factor,
            custom_audio_files=custom_audio_files
        )
        
        # Check if we have enough segments
        if len(empty_dataset) < num_empty_samples:
            print(f"WARNING: Requested {num_empty_samples} empty segments, but only found {len(empty_dataset)}.")
            num_empty_samples = len(empty_dataset)
        
        if num_empty_samples > 0:
            # Randomly select indices
            empty_indices = random.sample(range(len(empty_dataset)), num_empty_samples)
            
            # Save the selected segments
            for i, idx in enumerate(empty_indices):
                waveform, _ = empty_dataset[idx]
                
                # Ensure 2D tensor
                if waveform.ndim > 2:
                    waveform = waveform.squeeze(0)
                
                # Save as WAV file
                output_path = os.path.join(output_dir, f"empty_{i:04d}.wav")
                torchaudio.save(
                    output_path,
                    waveform,
                    sample_rate=target_sr
                )
                
                # Print progress
                if (i + 1) % 10 == 0 or i + 1 == num_empty_samples:
                    print(f"  Saved {i + 1}/{num_empty_samples} empty segments")
    
    # Count total saved files
    saved_files = len([f for f in os.listdir(output_dir) if f.endswith('.wav')])
    print(f"\nSaved {saved_files} 'no birds' samples to {output_dir}/")
    
    return saved_files
    
def main():
    """Parse arguments and run the sample generation."""
    parser = argparse.ArgumentParser(description="Generate 'no birds' class samples")
    parser.add_argument("--bird_dir", default="bird_sound_dataset", 
                        help="Directory containing bird sound recordings")
    parser.add_argument("--esc50_dir", default=None, 
                        help="Directory containing ESC-50 dataset (downloaded if not provided)")
    parser.add_argument("--output_dir", default="augmented_dataset", 
                        help="Directory to save the generated samples")
    parser.add_argument("--num_samples", type=int, default=100, 
                        help="Number of 'no birds' samples to generate")
    parser.add_argument("--esc50_ratio", type=float, default=0.5, 
                        help="Proportion of samples to source from ESC-50 (vs. empty segments)")
    parser.add_argument("--files_per_class", type=int, default=None, 
                        help="Maximum number of files to scan per bird class")
    parser.add_argument("--energy_threshold", type=float, default=1.5,
                        help="Energy threshold factor for detecting silence (higher values detect more segments)")
    
    args = parser.parse_args()
    
    # Check if bird directory exists
    if not os.path.exists(args.bird_dir):
        print(f"ERROR: Bird sound directory '{args.bird_dir}' not found.")
        return 1
    
    # Get ESC-50 directory if not provided
    esc50_dir = args.esc50_dir if args.esc50_dir else download_and_extract_esc50()
    
    # Generate and save the samples
    save_no_birds_samples(
        bird_data_dir=args.bird_dir,
        esc50_dir=esc50_dir,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        esc50_ratio=args.esc50_ratio,
        files_per_bird_class=args.files_per_class,
        energy_threshold_factor=args.energy_threshold
    )
    
    return 0

if __name__ == "__main__":
    main() 