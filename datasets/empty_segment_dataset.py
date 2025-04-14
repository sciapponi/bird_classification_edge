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

from .audio_utils import extract_empty_segments

class EmptySegmentDataset(Dataset):
    """Dataset for segments of audio with no bird calls."""
    
    def __init__(self, bird_data_dir, allowed_bird_classes, no_birds_label,
                 clip_duration=3.0, sr=22050, lowcut=2000, highcut=10000,
                 max_segments_per_file=5, energy_threshold_factor=0.5, custom_audio_files=None):
        """
        Initialize the Empty Segment Dataset.
        
        Args:
            bird_data_dir (str): Directory containing bird sound data
            allowed_bird_classes (list): List of bird classes (folder names) to scan
            no_birds_label (int): The integer label to assign to these empty segments
            clip_duration (float): Target duration for audio clips in seconds
            sr (int): Target sample rate for audio processing
            lowcut (int): Low frequency cutoff for bandpass filter
            highcut (int): High frequency cutoff for bandpass filter
            max_segments_per_file (int): Maximum number of segments to extract per file
            energy_threshold_factor (float): Factor for setting silence threshold
            custom_audio_files (list, optional): List of specific audio files to process
        """
        print("--- Initializing EmptySegmentDataset ---")
        self.bird_data_dir = bird_data_dir
        self.allowed_bird_classes = allowed_bird_classes
        self.no_birds_label = no_birds_label
        self.clip_duration = clip_duration
        self.sr = sr
        self.lowcut = lowcut
        self.highcut = highcut
        self.max_segments_per_file = max_segments_per_file
        self.energy_threshold_factor = energy_threshold_factor
        self.custom_audio_files = custom_audio_files

        self.segment_list = [] # Stores (file_path, start_time, end_time)

        self._find_empty_segments()
        print(f"--- EmptySegmentDataset initialized with {len(self.segment_list)} segments ---")

    def _find_empty_segments(self):
        """Find segments with no bird calls in the audio files."""
        if self.custom_audio_files is not None:
            # Use the provided list of audio files directly
            print(f"Scanning {len(self.custom_audio_files)} custom audio files for empty segments...")
            for audio_path in self.custom_audio_files:
                if os.path.exists(audio_path) and audio_path.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                    empty_intervals = extract_empty_segments(
                        audio_path,
                        clip_duration=self.clip_duration,
                        sr=self.sr,
                        lowcut=self.lowcut,
                        highcut=self.highcut,
                        energy_threshold_factor=self.energy_threshold_factor,
                        max_segments_per_file=self.max_segments_per_file
                    )
                    for start_time, end_time in empty_intervals:
                        self.segment_list.append((audio_path, start_time, end_time))
                else:
                    print(f"WARNING: File {audio_path} does not exist or is not a supported audio format")
        else:
            # Standard scanning through bird class directories
            for class_name in self.allowed_bird_classes:
                class_dir = os.path.join(self.bird_data_dir, class_name)
                if os.path.isdir(class_dir):
                    print(f"Scanning for empty segments in: {class_dir}")
                    for file in os.listdir(class_dir):
                        if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                            audio_path = os.path.join(class_dir, file)
                            empty_intervals = extract_empty_segments(
                                audio_path,
                                clip_duration=self.clip_duration,
                                sr=self.sr,
                                lowcut=self.lowcut,
                                highcut=self.highcut,
                                energy_threshold_factor=self.energy_threshold_factor,
                                max_segments_per_file=self.max_segments_per_file
                            )
                            for start_time, end_time in empty_intervals:
                                self.segment_list.append((audio_path, start_time, end_time))
                else:
                    print(f"WARNING: Folder {class_dir} does not exist while scanning for empty segments!")

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
            print(f"Error loading empty segment {idx} from {audio_path} ({start_time:.2f}-{end_time:.2f}): {e}")
            # Return a silent clip as fallback
            return torch.zeros(1, int(self.sr * self.clip_duration)), self.no_birds_label 