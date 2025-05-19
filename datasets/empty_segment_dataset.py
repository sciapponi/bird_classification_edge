"""
Empty Segment Dataset

This module provides a dataset implementation for extracting and using
segments of audio that contain no bird calls ("empty" segments).
"""

import os
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset
import random # Import random for shuffling

from .audio_utils import extract_empty_segments

class EmptySegmentDataset(Dataset):
    """Dataset for segments of audio with no bird calls."""

    # Class-level (static) variables to store the master pool
    _master_empty_segment_pool = []
    _pool_initialized = False
    _scan_in_progress = False # Lock to prevent concurrent scans if instantiating multiple times quickly (less likely here)

    def __init__(self, bird_data_dir, allowed_bird_classes, no_birds_label,
                 num_target_segments, # New parameter: how many segments this instance needs
                 clip_duration=3.0, sr=22050, lowcut=2000, highcut=10000,
                 max_segments_per_file=5, energy_threshold_factor=0.5, custom_audio_files=None,
                 seed=42): # Seed for shuffling the master pool and file list
        """
        Initialize the Empty Segment Dataset.
        
        Args:
            bird_data_dir (str): Directory containing bird sound data
            allowed_bird_classes (list): List of bird classes (folder names) to scan
            no_birds_label (int): The integer label to assign to these empty segments
            num_target_segments (int): Number of empty segments this dataset instance should provide.
            clip_duration (float): Target duration for audio clips in seconds
            sr (int): Target sample rate for audio processing
            lowcut (int): Low frequency cutoff for bandpass filter
            highcut (int): High frequency cutoff for bandpass filter
            max_segments_per_file (int): Maximum number of segments to extract per file
            energy_threshold_factor (float): Factor for setting silence threshold
            custom_audio_files (list, optional): List of specific audio files to process
            seed (int): Random seed for shuffling operations.
        """
        # print(f"--- Initializing EmptySegmentDataset instance (target: {num_target_segments}) ---")
        self.bird_data_dir = bird_data_dir
        self.allowed_bird_classes = allowed_bird_classes
        self.no_birds_label = no_birds_label
        self.num_target_segments = num_target_segments
        self.clip_duration = clip_duration
        self.sr = sr
        self.lowcut = lowcut
        self.highcut = highcut
        self.max_segments_per_file = max_segments_per_file
        self.energy_threshold_factor = energy_threshold_factor
        self.custom_audio_files = custom_audio_files # custom_audio_files will define the scope of the initial scan
        self.seed = seed

        self.segment_list = [] # Segments for this specific instance

        if not EmptySegmentDataset._pool_initialized and not EmptySegmentDataset._scan_in_progress:
            EmptySegmentDataset._scan_in_progress = True
            print("--- Populating master empty segment pool (one-time scan) ---")
            self._populate_master_pool()
            EmptySegmentDataset._pool_initialized = True
            EmptySegmentDataset._scan_in_progress = False
            print(f"--- Master empty segment pool populated with {len(EmptySegmentDataset._master_empty_segment_pool)} segments. ---")
        elif EmptySegmentDataset._scan_in_progress:
            # Should ideally wait or handle this, but for sequential train.py calls it's less of an issue.
            print("--- Scan for master pool already in progress, waiting for it to complete (this instance might be empty if called too soon) ---")


        self._get_segments_from_pool()
        
        # print(f"--- EmptySegmentDataset instance created with {len(self.segment_list)}/{self.num_target_segments} segments. Pool has {len(EmptySegmentDataset._master_empty_segment_pool)} remaining. ---")


    def _populate_master_pool(self):
        """Scans all relevant audio files ONCE to create a master list of all possible empty segments."""
        files_to_process = []
        if self.custom_audio_files is not None:
            # If custom_audio_files is provided, this defines the ENTIRE scope for empty segments for ALL dataset instances.
            # This might be desired if we want empty segments only from a specific split's files.
            print(f"Preparing to scan {len(self.custom_audio_files)} custom audio files for the master pool...")
            for audio_path in self.custom_audio_files:
                if os.path.exists(audio_path) and audio_path.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    files_to_process.append(audio_path)
                else:
                    print(f"WARNING (Master Pool Scan): Custom file {audio_path} does not exist or is not a supported audio format. Skipping.")
        else:
            # Standard scan: all allowed bird classes
            print("Preparing to scan all bird dataset directories for the master pool...")
            for class_name in self.allowed_bird_classes:
                class_dir = os.path.join(self.bird_data_dir, class_name)
                if os.path.isdir(class_dir):
                    for file_name in os.listdir(class_dir):
                        if file_name.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                            files_to_process.append(os.path.join(class_dir, file_name))
                else:
                    print(f"WARNING (Master Pool Scan): Class directory {class_dir} does not exist. Skipping.")
        
        total_files = len(files_to_process)
        if total_files == 0:
            print("WARNING (Master Pool Scan): No audio files found to scan for empty segments.")
            return

        # Shuffle files before processing to make segment order in pool somewhat random initially
        random.seed(self.seed)
        random.shuffle(files_to_process)

        print(f"Starting one-time master empty segment extraction from {total_files} files...")
        
        # Temporary list for this scan
        all_found_segments = []

        for idx, audio_path in enumerate(files_to_process):
            print(f"Scanning (Master Pool) {idx + 1}/{total_files}: {os.path.basename(audio_path)[:40]}...", end='\r')
            empty_intervals = extract_empty_segments(
                audio_path,
                clip_duration=self.clip_duration,
                sr=self.sr,
                lowcut=self.lowcut,
                highcut=self.highcut,
                energy_threshold_factor=self.energy_threshold_factor,
                max_segments_per_file=self.max_segments_per_file, # Get as many as possible per file
                verbose=False 
            )
            for start_time, end_time in empty_intervals:
                all_found_segments.append((audio_path, start_time, end_time))
        
        EmptySegmentDataset._master_empty_segment_pool.extend(all_found_segments)
        # Shuffle the final master pool
        random.seed(self.seed) # Re-seed for consistent shuffle of the pool itself
        random.shuffle(EmptySegmentDataset._master_empty_segment_pool)

        print("\n" + " " * 100 + "\r", end='') # Clears the progress line
        # Final message printed by the caller of _populate_master_pool

    def _get_segments_from_pool(self):
        num_to_take = self.num_target_segments
        
        if not EmptySegmentDataset._pool_initialized:
            # This case should ideally not be hit if logic in __init__ is correct
            # Or if multiple instances are created before the pool is ready.
            print(f"WARNING: Master pool not yet initialized. Cannot get {num_to_take} segments for this instance.")
            return

        if len(EmptySegmentDataset._master_empty_segment_pool) == 0:
            print(f"WARNING: Master empty segment pool is empty. Cannot provide {num_to_take} segments.")
            return

        if len(EmptySegmentDataset._master_empty_segment_pool) < num_to_take:
            print(f"WARNING: Master pool has only {len(EmptySegmentDataset._master_empty_segment_pool)} segments, "
                  f"but {num_to_take} were requested. Using all available.")
            num_to_take = len(EmptySegmentDataset._master_empty_segment_pool)
        
        # Take segments from the end of the list (pop) to ensure they are removed
        for _ in range(num_to_take):
            if EmptySegmentDataset._master_empty_segment_pool: # Check if still items left
                 self.segment_list.append(EmptySegmentDataset._master_empty_segment_pool.pop())
            else: # Should not happen if num_to_take logic is correct
                break 
        
        if len(self.segment_list) < self.num_target_segments:
             print(f"  INFO: EmptySegmentDataset instance for label {self.no_birds_label} received {len(self.segment_list)} out of {self.num_target_segments} requested segments.")


    def __len__(self):
        """Returns the total number of segments."""
        return len(self.segment_list)

    def __getitem__(self, idx):
        """
        Returns the audio segment and its label at the given index.
        
        Args:
            idx (int): Index of the segment
            
        Returns:
            tuple: (waveform, label)
        """
        if idx >= len(self.segment_list):
            # This can happen if the dataset is smaller than expected and DataLoader tries to access out of bounds
            # Or if _get_segments_from_pool didn't populate enough.
            print(f"ERROR: Attempting to access index {idx} in EmptySegmentDataset, but only {len(self.segment_list)} segments available.")
            # Fallback: return a silent tensor
            return torch.zeros(1, int(self.sr * self.clip_duration)), self.no_birds_label


        audio_path, start_time, end_time = self.segment_list[idx]

        try:
            # Load the specific segment using librosa
            waveform_np, loaded_sr = librosa.load(
                audio_path, 
                sr=self.sr, 
                offset=start_time, 
                duration=self.clip_duration
            )

            # Convert to torch tensor
            waveform = torch.from_numpy(waveform_np).float().unsqueeze(0) # Add channel dim

            # Ensure standard length 
            target_length_samples = int(self.sr * self.clip_duration)
            if waveform.shape[1] < target_length_samples:
                waveform = F.pad(waveform, (0, target_length_samples - waveform.shape[1]))
            elif waveform.shape[1] > target_length_samples:
                waveform = waveform[:, :target_length_samples]

            # Normalize audio
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            else:
                # It's possible to get a truly silent segment
                waveform = torch.zeros_like(waveform)

            return waveform, self.no_birds_label

        except Exception as e:
            print(f"Error loading empty segment {idx} (path: {audio_path}, time: {start_time:.2f}-{end_time:.2f}): {e}")
            # Return a silent clip as fallback
            return torch.zeros(1, int(self.sr * self.clip_duration)), self.no_birds_label 