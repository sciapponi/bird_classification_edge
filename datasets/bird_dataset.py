"""
Bird Sound Dataset

This module provides the dataset implementation for bird sound classification.
It includes functionality for call extraction and augmentation.
"""

import os
import random
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.sox_effects as sox_effects
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import math # Import math for floor/ceil

from .audio_utils import extract_call_segments

class BirdSoundDataset(Dataset):
    """Dataset class for bird sound classification."""
    
    def __init__(self, root_dir, transform=None, allowed_classes=[], subset="training", 
                 validation_split=0.15, test_split=0.15, split_seed=42, # Added split parameters
                 augment=False, preload=False,
                 extract_calls=True, clip_duration=3.0, sr=22050, lowcut=2000, highcut=10000,
                 background_dataset=None, custom_file_dict=None, verbose=False): # Added verbose
        """
        Initialize the Bird Sound Dataset.
        
        Args:
            root_dir (string): Directory with the Bird Sound dataset
            transform (callable, optional): Optional transform to be applied on a sample
            allowed_classes (list): List of bird classes (folder names) to include
            subset (string): Which subset to use ('training', 'validation', 'testing')
            validation_split (float): Proportion of data for the validation set (0.0-1.0)
            test_split (float): Proportion of data for the test set (0.0-1.0)
            split_seed (int): Random seed for reproducible data splitting
            augment (bool): Whether to apply augmentation to the data
            preload (bool): If True, preload all audio files into memory
            extract_calls (bool): If True, extract bird calls from audio files
            clip_duration (float): Duration of call clips in seconds
            sr (int): Sample rate for audio processing
            lowcut (int): Low frequency cutoff for bandpass filter
            highcut (int): High frequency cutoff for bandpass filter
            background_dataset (Dataset, optional): Dataset providing background noise samples for augmentation
            custom_file_dict (dict, optional): Dictionary mapping class names to lists of file paths to use
            verbose (bool): If True, print more detailed logs during initialization and item fetching.
        """
        # print(f"Initializing BirdSoundDataset: subset={subset}, augment={augment}, extract_calls={extract_calls}") # MODIFIED
        self.verbose = verbose
        if self.verbose:
            print(f"Initializing BirdSoundDataset: subset={subset}, augment={augment}, extract_calls={extract_calls}, verbose={self.verbose}")

        self.allowed_classes = allowed_classes
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.validation_split = validation_split
        self.test_split = test_split
        self.split_seed = split_seed
        self.augment = augment and subset == "training"
        self.preload = preload
        self.file_list = []
        self.labels = []
        self.class_to_idx = {}
        self.preloaded_data = {}
        self.custom_file_dict = custom_file_dict
        
        self.extract_calls = extract_calls
        self.clip_duration = clip_duration
        self.sr = sr
        self.lowcut = lowcut
        self.highcut = highcut
        self.background_dataset = background_dataset
        
        self._initialize_dataset()
        # print(f"Dataset initialized for subset '{self.subset}' with {len(self.file_list)} audio files") # MODIFIED
        if self.verbose or len(self.file_list) == 0 : # Print if verbose or if no files (warning)
             print(f"Dataset initialized for subset '{self.subset}' with {len(self.file_list)} audio files")
    
    def _initialize_dataset(self):
        """
        Initialize the dataset by finding all audio files, splitting them according 
        to the subset, and assigning labels.
        This method populates file_list and labels.
        """
        # print(f"Looking for audio files in {self.root_dir}") # MODIFIED
        # print(f"Allowed classes: {self.allowed_classes}") # MODIFIED
        if self.verbose:
            print(f"Looking for audio files in {self.root_dir}")
            print(f"Allowed classes: {self.allowed_classes}")
        
        if not self.allowed_classes:
            self.allowed_classes = [d for d in os.listdir(self.root_dir) 
                                  if os.path.isdir(os.path.join(self.root_dir, d)) and not d.startswith('.')]
            # print(f"No classes specified, using all folders: {self.allowed_classes}") # MODIFIED
            if self.verbose:
                 print(f"No classes specified, using all folders: {self.allowed_classes}")
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.allowed_classes)}
        
        files_per_class = {cls_name: [] for cls_name in self.allowed_classes}
        
        if self.custom_file_dict is not None:
            # print(f"Using custom file dictionary...") # MODIFIED
            if self.verbose: print(f"Using custom file dictionary...")
            for class_name, file_paths in self.custom_file_dict.items():
                if class_name in self.class_to_idx:
                    files_per_class[class_name].extend(file_paths)
                    # print(f"  Collected {len(file_paths)} files for class {class_name} from dict") # MODIFIED
                    if self.verbose: print(f"  Collected {len(file_paths)} files for class {class_name} from dict")
                else:
                    print(f"WARNING: Class {class_name} in custom_file_dict not in allowed_classes, skipping")
        else:
            for class_name in self.allowed_classes:
                class_dir = os.path.join(self.root_dir, class_name)
                if os.path.isdir(class_dir):
                    # print(f"Scanning folder: {class_dir}") # MODIFIED
                    if self.verbose: print(f"Scanning folder: {class_dir}")
                    count = 0
                    for file in os.listdir(class_dir):
                        if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                            files_per_class[class_name].append(os.path.join(class_dir, file))
                            count += 1
                    # print(f"  Collected {count} audio files in {class_name}") # MODIFIED
                    if self.verbose: print(f"  Collected {count} audio files in {class_name}")
                else:
                    print(f"WARNING: Folder {class_dir} does not exist!")

        # print(f"Splitting data using seed {self.split_seed}, val_split={self.validation_split}, test_split={self.test_split}") # MODIFIED
        if self.verbose:
            print(f"Splitting data using seed {self.split_seed}, val_split={self.validation_split}, test_split={self.test_split}")
        self.file_list = []
        self.labels = []    
        
        rng = random.Random(self.split_seed)

        for class_name, all_files in files_per_class.items():
            if not all_files:
                print(f"WARNING: No files found for class {class_name}, skipping.")
                continue

            num_files = len(all_files)
            # print(f"  Processing class '{class_name}' with {num_files} files.") # MODIFIED
            if self.verbose: print(f"  Processing class '{class_name}' with {num_files} files.")
            
            rng.shuffle(all_files) 
            
            num_val = math.floor(self.validation_split * num_files)
            num_test = math.floor(self.test_split * num_files)
            num_train = num_files - num_val - num_test

            if num_train <= 0 or num_val < 0 or num_test < 0:
                 print(f"WARNING: Invalid split sizes for class {class_name} (train={num_train}, val={num_val}, test={num_test}). Check split percentages.")
                 if self.subset == 'training':
                     selected_files = all_files
                 else:
                     selected_files = []
            else:
                # print(f"    Split counts: Train={num_train}, Val={num_val}, Test={num_test}") # MODIFIED
                if self.verbose: print(f"    Split counts: Train={num_train}, Val={num_val}, Test={num_test}")
                
                if self.subset == 'training':
                    selected_files = all_files[:num_train]
                elif self.subset == 'validation':
                    selected_files = all_files[num_train : num_train + num_val]
                elif self.subset == 'testing':
                    selected_files = all_files[num_train + num_val :]
                else:
                    raise ValueError(f"Invalid subset value: {self.subset}. Must be 'training', 'validation', or 'testing'.")

            self.file_list.extend(selected_files)
            self.labels.extend([self.class_to_idx[class_name]] * len(selected_files))
            # print(f"    Added {len(selected_files)} files for subset '{self.subset}'") # MODIFIED
            if self.verbose: print(f"    Added {len(selected_files)} files for subset '{self.subset}' for class '{class_name}'")


        if not self.file_list:
             print(f"WARNING: No files selected for the subset '{self.subset}'. Check configuration and data.")

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        
        # print(f"Loading sample {idx}: {audio_path}") # REMOVED
        
        loaded_from_cache = False
        if self.preload and audio_path in self.preloaded_data:
            # print("Using preloaded data") # REMOVED
            waveform = self.preloaded_data[audio_path]
            loaded_from_cache = True
        
        if not loaded_from_cache:
            if self.extract_calls:
                # print("Extracting calls...") # REMOVED
                call_intervals, segments, _, _, _ = extract_call_segments(
                    audio_path, 
                    clip_duration=self.clip_duration,
                    sr=self.sr, 
                    lowcut=self.lowcut, 
                    highcut=self.highcut,
                    verbose=self.verbose # Pass verbose to audio_utils
                )
                
                if segments and len(segments) > 0:
                    # print(f"Using the first of {len(segments)} extracted segments") # REMOVED
                    # Randomly select one of the extracted segments if multiple are found
                    # This introduces more variability during training
                    # For validation/testing, consistently using the first one might be better for reproducibility
                    # if self.subset == "training" and len(segments) > 1:
                    #    selected_segment_idx = random.randint(0, len(segments) - 1)
                    # else:
                    #    selected_segment_idx = 0 
                    # audio_data = segments[selected_segment_idx] 
                    
                    # For now, let's stick to the first segment for simplicity and consistency
                    audio_data = segments[0] 
                    waveform = torch.from_numpy(audio_data).float()
                    
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                    
                    waveform = self.ensure_length(waveform)

                else:
                    # print("No calls detected, using standard loading") # REMOVED
                    if self.verbose:
                        print(f"No calls detected in {audio_path} by __getitem__, using standard loading.")
                    waveform = self.load_audio(audio_path)
            else:
                # print("Standard audio loading (without call extraction)") # REMOVED
                waveform = self.load_audio(audio_path)
            
            if self.preload and not loaded_from_cache:
                self.preloaded_data[audio_path] = waveform
        
        if self.augment:
            # print("Applying augmentation...") # REMOVED
            waveform = self.apply_augmentation(waveform) 
        
        if self.transform:
            # print("Applying transform...") # REMOVED
            waveform = self.transform(waveform)
        
        # print(f"Sample ready: shape={waveform.shape}, label={label}") # REMOVED
        return waveform, label
    
    def load_audio(self, audio_path):
        """
        Load an audio file and process it to a standard format.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Processed audio waveform
        """
        try:
            waveform, sr_orig = torchaudio.load(audio_path)
        except Exception as e:
            print(f"ERROR loading audio file {audio_path}: {e}. Returning zeros.")
            # Return a zero tensor of the expected shape if loading fails
            # Expected shape is [channels, num_samples]
            # Assuming mono audio for now, and target_length is sr * clip_duration
            target_len_samples = int(self.sr * self.clip_duration)
            return torch.zeros((1, target_len_samples))

        # Resample if necessary
        if sr_orig != self.sr:
            # print(f"Resampling {audio_path} from {sr_orig}Hz to {self.sr}Hz") # MODIFIED
            if self.verbose: print(f"Resampling {audio_path} from {sr_orig}Hz to {self.sr}Hz")
            resampler = T.Resample(sr_orig, self.sr)
            waveform = resampler(waveform)
            
        # Convert to mono if necessary by averaging channels
        if waveform.shape[0] > 1:
            # print(f"Converting {audio_path} to mono.") # MODIFIED
            if self.verbose: print(f"Converting {audio_path} to mono.")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ensure correct length (padding or truncating)
        waveform = self.ensure_length(waveform)
        
        return waveform
    
    def apply_augmentation(self, waveform):
        """
        Apply random augmentations to a waveform.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            
        Returns:
            torch.Tensor: Augmented waveform
        """
        aug_transforms = [
            self.add_noise,
            self.time_mask,
            self.freq_mask,
            self.time_shift,
            self.speed_perturb,
            # self.reverb # Reverb can be computationally expensive
        ]

        # Add mixing if background data is available
        if self.background_dataset is not None:
            aug_transforms.append(self.mix_with_background)

        # Apply a random selection of augmentations
        random.shuffle(aug_transforms)
        for aug in aug_transforms[:2]:  # Apply at most 2 augmentations
            if random.random() < 0.5:  # 50% chance to apply each selected augmentation
                waveform = aug(waveform)
        
        # Ensure proper length after augmentation
        waveform = self.ensure_length(waveform)
        
        return waveform
        
    def ensure_length(self, waveform, target_length=None):
        """
        Ensure the waveform is exactly the target length by padding or trimming.
        
        Args:
            waveform (torch.Tensor): Input audio waveform.
                                     Expected shapes: [channels, time], [1, channels, time], or [time].
            target_length (int, optional): The target length in samples. 
                                         If None, uses self.sr * self.clip_duration.
        
        Returns:
            torch.Tensor: Waveform of the target length, shape [channels, target_length].
                          Returns a silent clip [1, target_length] on error.
        """
        if target_length is None:
            target_length = int(self.sr * self.clip_duration)

        # Standardize waveform shape to [channels, time]
        if waveform.ndim == 3:
            if waveform.shape[0] == 1: # Shape [1, channels, time] -> [channels, time]
                waveform = waveform.squeeze(0)
            elif waveform.shape[1] == 1: # Shape [channels, 1, time] -> [channels, time] (improbabile ma gestito)
                waveform = waveform.squeeze(1)
            else: # Forma 3D non riconosciuta (es. [batch>1, canali, tempo])
                print(f"WARNING in ensure_length: Unhandled 3D waveform shape {waveform.shape}. Using first element if batch-like or returning zeros.")
                # Se sembra un batch, proviamo a prendere il primo elemento
                if waveform.shape[0] > 1 and waveform.ndim == 3 : # Esempio [batch, channels, time]
                     print(f"Assuming batch-like shape, taking waveform[0]...")
                     waveform = waveform[0] 
                     if waveform.ndim == 1: # Se diventa [time]
                          waveform = waveform.unsqueeze(0) # -> [1, time]
                else:
                    print(f"ERROR in ensure_length: Could not standardize 3D shape {waveform.shape} to [channels, time]. Returning zeros.")
                    return torch.zeros(1, target_length)
        
        if waveform.ndim == 1: # Shape [time] -> [1, time]
            waveform = waveform.unsqueeze(0)
        
        # At this point, waveform should be [channels, time]
        if waveform.ndim != 2:
            print(f"ERROR in ensure_length: Waveform shape {waveform.shape} is not 2D after standardization. Returning zeros.")
            return torch.zeros(1, target_length)

        # Proceed with trimming or padding
        num_channels = waveform.shape[0]
        current_length = waveform.shape[1]
        
        if current_length == target_length:
            return waveform
        elif current_length > target_length:
            # Trim
            return waveform[:, :target_length]
        else: # current_length < target_length
            # Pad
            padding_needed = target_length - current_length
            # Pad on the right (at the end of the clip)
            # F.pad expects (pad_left, pad_right, pad_top, pad_bottom, ...)
            # For a 2D tensor [channels, time], we only pad the last dimension (time)
            return F.pad(waveform, (0, padding_needed))
    
    def add_noise(self, waveform, noise_level=0.005):
        """
        Add random Gaussian noise to the waveform.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            noise_level (float): Standard deviation of the noise
            
        Returns:
            torch.Tensor: Noisy waveform
        """
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def time_mask(self, waveform, time_mask_param=20):
        """
        Apply time masking augmentation.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            time_mask_param (int): Maximum possible length of the mask
            
        Returns:
            torch.Tensor: Time-masked waveform
        """
        transform = T.TimeMasking(time_mask_param=time_mask_param)
        return transform(waveform)
    
    def freq_mask(self, waveform, freq_mask_param=10):
        """
        Apply frequency masking augmentation.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            freq_mask_param (int): Maximum possible length of the mask
            
        Returns:
            torch.Tensor: Frequency-masked waveform
        """
        # Convert to spectrogram
        spec = torchaudio.transforms.Spectrogram()(waveform)
        # Apply frequency masking
        transform = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        masked_spec = transform(spec)
        # Convert back to waveform 
        # This is an approximation, as perfect inversion is not always possible
        griffin_lim = torchaudio.transforms.GriffinLim()(masked_spec)
        return griffin_lim.unsqueeze(0)
    
    def time_shift(self, waveform, shift_limit=0.2):
        """
        Shift the waveform in time.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            shift_limit (float): Maximum shift as a fraction of waveform length
            
        Returns:
            torch.Tensor: Time-shifted waveform
        """
        shift = int(shift_limit * waveform.shape[1] * (random.random() - 0.5))  
        return torch.roll(waveform, shifts=shift, dims=1)
        
    def speed_perturb(self, waveform, rate_min=0.9, rate_max=1.1):
        """
        Change the speed of the waveform.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            rate_min (float): Minimum speed rate
            rate_max (float): Maximum speed rate
            
        Returns:
            torch.Tensor: Speed-perturbed waveform
        """
        rate = random.uniform(rate_min, rate_max)
        orig_len = waveform.shape[1]
        
        # Interpolate to change speed
        new_len = int(orig_len / rate)
        waveform_np = waveform.numpy().squeeze()
        
        # Make sure time_orig matches the actual length of waveform_np
        time_orig = np.arange(len(waveform_np))
        time_new = np.linspace(0, len(waveform_np)-1, new_len)
        
        waveform_speed = np.interp(time_new, time_orig, waveform_np)
        
        return torch.from_numpy(waveform_speed).float().unsqueeze(0)
        
    def mix_with_background(self, waveform, snr_db_min=5, snr_db_max=15):
        """
        Mix the waveform with a random background sound from the background dataset.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            snr_db_min (float): Minimum signal-to-noise ratio in dB
            snr_db_max (float): Maximum signal-to-noise ratio in dB
            
        Returns:
            torch.Tensor: Mixed waveform
        """
        if self.background_dataset is None or len(self.background_dataset) == 0:
            print("WARN: Background dataset not provided or empty, skipping mix augmentation.")
            return waveform

        # Get a random background sample
        bg_idx = random.randint(0, len(self.background_dataset) - 1)
        # Assuming background_dataset returns (waveform, label)
        bg_waveform, _ = self.background_dataset[bg_idx]

        # Ensure background waveform is loaded and mono
        # (ESC50Dataset should already handle this, but double-check)
        if bg_waveform.shape[0] > 1:
             bg_waveform = bg_waveform.mean(dim=0, keepdim=True)
        if bg_waveform.shape[1] == 0: # Handle potential loading errors
             print("WARN: Empty background waveform loaded, skipping mix.")
             return waveform

        # Ensure background is long enough, tile if necessary
        target_len = waveform.shape[1]
        while bg_waveform.shape[1] < target_len:
            bg_waveform = torch.cat([bg_waveform, bg_waveform], dim=1)
            
        # Trim if too long
        if bg_waveform.shape[1] > target_len:
            start = random.randint(0, bg_waveform.shape[1] - target_len)
            bg_waveform = bg_waveform[:, start:start + target_len]

        # Choose random SNR
        snr_db = random.uniform(snr_db_min, snr_db_max)
        snr_linear = 10**(snr_db / 10.0)

        # Calculate power (RMS squared)
        # Add small epsilon to prevent division by zero
        epsilon = 1e-10
        power_signal = torch.mean(waveform**2) + epsilon
        power_noise = torch.mean(bg_waveform**2) + epsilon

        # Calculate scaling factor for noise
        scale_factor = torch.sqrt(power_signal / (power_noise * snr_linear))

        # Mix
        mixed_waveform = waveform + scale_factor * bg_waveform

        # Normalize the mixed waveform (peak normalization)
        if mixed_waveform.abs().max() > 0:
             mixed_waveform = mixed_waveform / mixed_waveform.abs().max()
        else: # Handle case of pure silence
             mixed_waveform = torch.zeros_like(waveform)

        # Opzionale: stampa l'SNR per il debug
        # print(f"Applied background mix: SNR={snr_db:.2f}dB")

        return mixed_waveform
        
    def reverb(self, waveform, sample_rate=None, reverberance=50):
        """
        Apply reverberation effect to the waveform.
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            sample_rate (int, optional): Sample rate of the audio
            reverberance (int): Reverberance strength (0-100)
            
        Returns:
            torch.Tensor: Reverberated waveform
        """
        if sample_rate is None:
            sample_rate = self.sr
            
        effects = [
            ["reverb", str(reverberance)]  # Apply reverb with given strength (0-100)
        ]
        waveform_processed, _ = sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        return waveform_processed 

    # Helper methods for inspection
    def get_class_counts(self):
        """Returns a dictionary mapping class index to sample count."""
        counts = {}
        # Ensure labels are integers if they are not already
        int_labels = [int(label) for label in self.labels]
        for label in int_labels:
            counts[label] = counts.get(label, 0) + 1
        return counts

    def get_num_classes(self):
        """Returns the number of bird classes managed by this dataset instance."""
        return len(self.class_to_idx)

    def get_class_to_idx(self):
        """Returns the class_to_idx mapping."""
        return self.class_to_idx 