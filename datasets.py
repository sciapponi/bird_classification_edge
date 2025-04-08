import os 
import requests
import tarfile
import random
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
import torchaudio.sox_effects as sox_effects 
import torch
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import find_peaks, butter, filtfilt
from pathlib import Path
import pandas as pd
import time

# ALLOWED_CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

# Utility functions for bird call extraction
def butter_bandpass(lowcut, highcut, fs, order=4):
    """Create a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def compute_adaptive_parameters(y, sr, lowcut, highcut):
    """
    Compute adaptive parameters for bird call detection based on the overall filtered energy.
    Returns adaptive prominence for peak detection and an adaptive energy threshold.
    """
    # Filter the audio to the bird call frequency band.
    y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
    
    # Compute a short-time amplitude envelope using a moving average.
    frame_length = int(sr * 0.05)  # 50 ms window
    hop_length = int(sr * 0.01)    # 10 ms hop
    envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
    envelope = envelope_frames.mean(axis=0)
    
    # Use the median and MAD (median absolute deviation) as robust measures.
    median_env = np.median(envelope)
    mad_env = np.median(np.abs(envelope - median_env))
    
    # Set the adaptive prominence: median + k * MAD (tune k as needed)
    adaptive_prominence = median_env + 1.5 * mad_env

    # Compute overall RMS energy of the filtered signal.
    rms_all = np.sqrt(np.mean(y_filtered ** 2))
    # Set the adaptive energy threshold (this is the baseline, and will be lowered for background verification).
    adaptive_energy_threshold = 0.5 * rms_all

    return adaptive_prominence, adaptive_energy_threshold

def extract_call_segments(audio_path, output_folder=None, clip_duration=3.0, sr=22050,
                         lowcut=2000, highcut=10000, min_peak_distance=1.0, height_percentile=75,
                         verbose=False, save_clips=False):
    """
    Detects bird calls by applying a bandpass filter and using an adaptive threshold
    for peak detection, then extracts clips centered on detected peaks.
    
    Returns a list of call intervals as (start_time, end_time), along with the audio data, sample rate, and duration.
    """
    print(f"Processing file: {audio_path}")
    start_time = time.time()
    
    # Create output folder if needed
    if save_clips and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(audio_path):
        if verbose:
            print(f"File not found: {audio_path}")
        return [], None, None, None, 0
    
    try:
        # Load audio with downsampling if needed
        print(f"Loading audio with librosa...")
        y, sr = librosa.load(audio_path, sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Audio loaded: {duration:.2f} seconds, sr={sr}")
    
        # Get the filename without extension for naming saved clips
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Compute adaptive parameters for detection.
        print("Calculating adaptive parameters...")
        adaptive_prominence, _ = compute_adaptive_parameters(y, sr, lowcut, highcut)
        
        # Filter the full audio for detection.
        print(f"Applying bandpass filter {lowcut}-{highcut} Hz...")
        y_filtered = apply_bandpass_filter(y, lowcut, highcut, sr, order=4)
        
        # Compute amplitude envelope from the filtered signal.
        print("Computing envelope...")
        frame_length = int(sr * 0.05)
        hop_length = int(sr * 0.01)
        envelope_frames = librosa.util.frame(np.abs(y_filtered), frame_length=frame_length, hop_length=hop_length)
        envelope = envelope_frames.mean(axis=0)
        
        # Calculate minimum distance between peaks in frames
        min_peak_distance_frames = int(min_peak_distance / (hop_length / sr))
        
        # Detect peaks using the adaptive prominence and minimum distance
        print("Detecting peaks...")
        peaks, properties = find_peaks(envelope, 
                                    prominence=adaptive_prominence,
                                    distance=min_peak_distance_frames)
        
        # Handle case with no peaks detected
        if len(peaks) == 0:
            print("No peaks detected!")
            if verbose:
                print(f"No peaks detected in {audio_path}. Try adjusting detection parameters.")
            return [], [], y, sr, duration
        
        print(f"Detected {len(peaks)} raw peaks")
        
        # Sort peaks by prominence (highest first)
        sorted_indices = np.argsort(-properties['prominences'])
        sorted_peaks = peaks[sorted_indices]
        sorted_prominences = properties['prominences'][sorted_indices]
        
        # Keep only the top percentile of peaks based on height/amplitude
        if len(sorted_peaks) > 0:  # Check if any peaks were found
            height_threshold = np.percentile(envelope[sorted_peaks], height_percentile)
            selected_peaks = [p for i, p in enumerate(sorted_peaks) if envelope[p] >= height_threshold]
        else:
            selected_peaks = []
        
        # Convert peaks to time
        peak_times = librosa.frames_to_time(selected_peaks, sr=sr, hop_length=hop_length)
        
        print(f"Selected {len(peak_times)} significant peaks")
        if verbose:
            print(f"Detected {len(peak_times)} significant bird calls in {audio_path}")

        call_intervals = []
        segments = []
        for i, t in enumerate(peak_times):
            start_time_sec = max(0, t - clip_duration / 2)  # Ensure we don't go below 0
            end_time_sec = min(duration, t + clip_duration / 2)  # Ensure we don't exceed audio length

            start_sample = int(start_time_sec * sr)
            end_sample = int(end_time_sec * sr)
            segment = y[start_sample:end_sample]
            segments.append(segment)

            # Save the clip if requested
            if save_clips and output_folder:
                filename = os.path.join(output_folder, f"{base_filename}_call_{i+1:03d}.wav")
                sf.write(filename, segment, sr)
                if verbose:
                    print(f"Saved call clip: {filename}")

            call_intervals.append((start_time_sec, end_time_sec))
        
        processing_time = time.time() - start_time
        print(f"Extracted {len(segments)} audio segments in {processing_time:.2f} seconds")
        return call_intervals, segments, y, sr, duration
    except Exception as e:
        print(f"ERROR processing {audio_path}: {str(e)}")
        if verbose:
            print(f"Error processing {audio_path}: {str(e)}")
        return [], [], None, None, 0    

def download_and_extract_bird_dataset():
    """
    Downloads and extracts the Bird Sound dataset
    """
    pass

def download_and_extract_esc50():
    """
    Downloads and extracts the ESC-50 dataset if it doesn't exist
    
    Returns:
        str: Path to the extracted dataset
    """
    # URLs and file paths
    esc50_url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    esc50_zip = "esc-50/ESC-50-master.zip"
    esc50_dir = "esc-50/ESC-50-master"
    
    # Check if the dataset already exists
    if os.path.exists(esc50_dir):
        print("ESC-50 dataset already exists.")
        return esc50_dir
    
    # Download the dataset
    print("Downloading ESC-50 dataset...")
    import requests
    response = requests.get(esc50_url, stream=True)
    with open(esc50_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

    # Extract the dataset
    print("Extracting ESC-50 dataset...")
    import zipfile
    with zipfile.ZipFile(esc50_zip, "r") as zip_ref:
        zip_ref.extractall(".")
    
    # Clean up
    os.remove(esc50_zip)
    
    print("ESC-50 dataset downloaded and extracted.")
    return esc50_dir

class ESC50Dataset(Dataset):
    """
    Environmental Sound Classification Dataset (ESC-50)
    Used to provide non-bird sound samples as negative examples
    """
    def __init__(self, root_dir, fold=None, exclude_bird_classes=True, 
                transform=None, target_sr=22050, target_length=3.0):
        """
        Args:
            root_dir (str): Root directory of ESC-50 dataset
            fold (int or list, optional): Which fold(s) to use (1-5), None for all
            exclude_bird_classes (bool): Whether to exclude bird/animal sounds
            transform (callable, optional): Transform to apply to audio
            target_sr (int): Target sample rate
            target_length (float): Target audio length in seconds
        """
        self.root_dir = root_dir
        self.audio_dir = os.path.join(root_dir, 'audio')
        self.meta_file = os.path.join(root_dir, 'meta', 'esc50.csv')
        self.transform = transform
        self.target_sr = target_sr
        self.target_length = target_length
        
        # Load metadata
        self.df = pd.read_csv(self.meta_file)
        
        # Filter by fold if specified
        if fold is not None:
            if isinstance(fold, list):
                self.df = self.df[self.df['fold'].isin(fold)]
            else:
                self.df = self.df[self.df['fold'] == fold]
        
        # Exclude bird and animal classes if specified
        if exclude_bird_classes:
            # Categories to exclude: animals, birds, etc.
            exclude_categories = ['animals', 'natural_soundscapes']
            self.df = self.df[~self.df['category'].isin(exclude_categories)]
        
        # Create a mapping of category to index
        self.categories = sorted(self.df['category'].unique())
        self.category_to_idx = {category: i for i, category in enumerate(self.categories)}
        
        # Create file list
        self.file_list = [os.path.join(self.audio_dir, filename) for filename in self.df['filename']]
        self.labels = [self.category_to_idx[category] for category in self.df['category']]
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        waveform = self.load_audio(audio_path)
        
        # Apply transform if specified
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label
    
    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.target_sr)
                waveform = resampler(waveform)
            
            # Ensure standard length
            target_length_samples = int(self.target_sr * self.target_length)
            
            if waveform.shape[1] < target_length_samples:
                # Pad if shorter
                waveform = F.pad(waveform, (0, target_length_samples - waveform.shape[1]))
            else:
                # Randomly crop if longer
                start = random.randint(0, waveform.shape[1] - target_length_samples)
                waveform = waveform[:, start:start + target_length_samples]
            
            # Normalize audio
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            return waveform
        
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a silent clip as fallback
            return torch.zeros(1, int(self.target_sr * self.target_length))

class BirdSoundDataset(Dataset):
    def __init__(self, root_dir, transform=None, allowed_classes=[], subset="training", augment=False, preload=False,
                extract_calls=True, clip_duration=3.0, sr=22050, lowcut=2000, highcut=10000):
        """
        Bird Sound Dataset
        
        Args:
            root_dir (string): Directory with the Bird Sound dataset
            transform (callable, optional): Optional transform to be applied on a sample
            subset (string): Which subset to use ('training', 'validation', 'testing')
            augment (bool): Whether to apply augmentation to the data
            preload (bool): If True, preload all audio files into memory
            extract_calls (bool): If True, extract bird calls from audio files
            clip_duration (float): Duration of call clips in seconds
            sr (int): Sample rate for audio processing
            lowcut (int): Low frequency cutoff for bandpass filter
            highcut (int): High frequency cutoff for bandpass filter
        """
        print(f"Initializing BirdSoundDataset: subset={subset}, augment={augment}, extract_calls={extract_calls}")
        self.allowed_classes = allowed_classes
        self.root_dir = root_dir
        self.transform = transform
        self.subset = subset
        self.augment = augment and subset == "training"  # Only augment training data
        self.preload = preload
        self.file_list = []
        self.labels = []
        self.class_to_idx = {}
        self.preloaded_data = {}
        
        # Bird call extraction parameters
        self.extract_calls = extract_calls
        self.clip_duration = clip_duration
        self.sr = sr
        self.lowcut = lowcut
        self.highcut = highcut
        
        # Initialize dataset (this would be implemented to load file paths and labels)
        self._initialize_dataset()
        print(f"Dataset initialized with {len(self.file_list)} audio files")
    
    def _initialize_dataset(self):
        """
        Initialize the dataset by finding all audio files and their labels.
        This method should populate file_list and labels based on the directory structure.
        """
        # Example implementation - replace with actual data loading logic
        print(f"Looking for audio files in {self.root_dir}")
        print(f"Allowed classes: {self.allowed_classes}")
        
        # If allowed_classes is empty, use all folders as classes
        if not self.allowed_classes:
            self.allowed_classes = [d for d in os.listdir(self.root_dir) 
                                  if os.path.isdir(os.path.join(self.root_dir, d))]
            print(f"No classes specified, using all folders: {self.allowed_classes}")
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.allowed_classes)}
        
        for class_name in self.allowed_classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                print(f"Scanning folder: {class_dir}")
                audio_files = []
                for file in os.listdir(class_dir):
                    if file.endswith(('.wav', '.mp3', '.ogg', '.flac')):
                        audio_files.append(file)
                        self.file_list.append(os.path.join(class_dir, file))
                        self.labels.append(self.class_to_idx[class_name])
                print(f"  Found {len(audio_files)} audio files in {class_name}")
            else:
                print(f"WARNING: Folder {class_dir} does not exist!")
    
    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """Returns a sample from the dataset at the given index"""
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        
        print(f"Loading sample {idx}: {audio_path}")
        
        if self.preload and audio_path in self.preloaded_data:
            print("Using preloaded data")
            waveform = self.preloaded_data[audio_path]
        else:
            if self.extract_calls:
                # Use the bird call extraction to get meaningful segments
                print("Extracting calls...")
                call_intervals, segments, _, _, _ = extract_call_segments(
                    audio_path, 
                    clip_duration=self.clip_duration,
                    sr=self.sr, 
                    lowcut=self.lowcut, 
                    highcut=self.highcut
                )
                
                if segments and len(segments) > 0:
                    # Use the first extracted call segment
                    print(f"Using the first of {len(segments)} extracted segments")
                    audio_data = segments[0]
                    waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
                else:
                    # Fallback to standard loading if no calls detected
                    print("No calls detected, using standard loading")
                    waveform = self.load_audio(audio_path)
            else:
                # Standard audio loading without call extraction
                print("Standard audio loading (without call extraction)")
                waveform = self.load_audio(audio_path)
            
            if self.preload:
                self.preloaded_data[audio_path] = waveform
        
        # Apply augmentation if enabled
        if self.augment:
            print("Applying augmentation...")
            waveform = self.apply_augmentation(waveform)
        
        # Apply additional transform if provided
        if self.transform:
            print("Applying transform...")
            waveform = self.transform(waveform)
        
        print(f"Sample ready: shape={waveform.shape}, label={label}")
        return waveform, label
    
    def load_audio(self, audio_path):
        """
        Loads an audio file and processes it to a standard format
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Processed waveform
        """
        # Load audio file using torchaudio
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.sr:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sr)
                waveform = resampler(waveform)
            
            # Ensure standard length
            waveform = self.ensure_length(waveform, self.sr * self.clip_duration)
            
            # Normalize audio
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
                
            return waveform
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a silent clip of the proper length as fallback
            return torch.zeros(1, int(self.sr * self.clip_duration))
    
    def apply_augmentation(self, waveform):
        """
        Apply random augmentations to a waveform
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Augmented waveform
        """
        aug_transforms = [
            self.add_noise,
            self.time_mask,
            self.freq_mask,
            self.time_shift,
            self.speed_perturb
        ]
        
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
        Ensures the waveform is exactly the target length
        
        Args:
            waveform: Input audio waveform
            target_length: Desired length in samples
            
        Returns:
            Waveform of exactly target_length samples
        """
        if target_length is None:
            target_length = int(self.sr * self.clip_duration)
            
        num_samples = waveform.shape[1]
        
        # Pad if shorter than the target length
        if num_samples < target_length:
            waveform = F.pad(waveform, (0, target_length - num_samples))
        # Trim if longer than the target length
        elif num_samples > target_length:
            waveform = waveform[:, :target_length]
        
        return waveform
    
    def add_noise(self, waveform, noise_level=0.005):
        """Adds random noise to the waveform"""
        noise = torch.randn_like(waveform) * noise_level
        return waveform + noise
    
    def time_mask(self, waveform, time_mask_param=20):
        """Applies time masking augmentation"""
        transform = T.TimeMasking(time_mask_param=time_mask_param)
        return transform(waveform)
    
    def freq_mask(self, waveform, freq_mask_param=10):
        """Applies frequency masking augmentation"""
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
        """Shifts the waveform in time"""
        shift = int(shift_limit * waveform.shape[1] * (random.random() - 0.5))  
        return torch.roll(waveform, shifts=shift, dims=1)

    def speed_perturb(self, waveform, rate_min=0.9, rate_max=1.1):
        """Changes the speed of the waveform"""
        rate = random.uniform(rate_min, rate_max)
        orig_len = waveform.shape[1]
        
        # Interpolate to change speed
        new_len = int(orig_len / rate)
        waveform_np = waveform.numpy().squeeze()
        time_orig = np.arange(orig_len)
        time_new = np.linspace(0, orig_len-1, new_len)
        waveform_speed = np.interp(time_new, time_orig, waveform_np)
        
        return torch.from_numpy(waveform_speed).float().unsqueeze(0)

    def reverb(self, waveform, sample_rate=None, reverberance=50):
        """Applies reverberation effect to the waveform"""
        if sample_rate is None:
            sample_rate = self.sr
            
        effects = [
            ["reverb", str(reverberance)]  # Apply reverb with given strength (0-100)
        ]
        waveform_processed, _ = sox_effects.apply_effects_tensor(waveform, sample_rate, effects)
        return waveform_processed

def create_combined_dataset(bird_data_dir, esc50_dir, allowed_bird_classes=None, 
                          bird_to_background_ratio=2.0, use_augmentation=True, 
                          target_sr=22050, clip_duration=3.0, subset="training"):
    """
    Create a combined dataset of bird sounds and ESC-50 background sounds
    
    Args:
        bird_data_dir (str): Directory containing bird sound data
        esc50_dir (str): Directory containing ESC-50 dataset
        allowed_bird_classes (list): List of bird classes to include (None for all)
        bird_to_background_ratio (float): Ratio of bird samples to background samples
        use_augmentation (bool): Whether to use augmentation
        target_sr (int): Target sample rate
        clip_duration (float): Clip duration in seconds
        subset (str): Which subset to use ("training", "validation", "testing")
        
    Returns:
        ConcatDataset: Combined dataset
    """
    print(f"Creating combined dataset: {subset}")
    # Set up transforms (could be more complex)
    transform = None
    
    # Create bird dataset
    print(f"Creating bird dataset from {bird_data_dir}...")
    bird_dataset = BirdSoundDataset(
        root_dir=bird_data_dir,
        transform=transform,
        allowed_classes=allowed_bird_classes,
        subset=subset,
        augment=use_augmentation and subset == "training",
        sr=target_sr,
        clip_duration=clip_duration,
        extract_calls=True
    )
    
    # Calculate how many ESC-50 samples we need based on the ratio
    num_bird_samples = len(bird_dataset)
    num_background_samples = int(num_bird_samples / bird_to_background_ratio)
    
    # For ESC-50, use different folds for different subsets
    if subset == "training":
        folds = [1, 2, 3]
    elif subset == "validation":
        folds = [4]
    else:  # testing
        folds = [5]
    
    # Create ESC-50 dataset
    esc50_dataset = ESC50Dataset(
        root_dir=esc50_dir,
        fold=folds,
        exclude_bird_classes=True,
        transform=transform,
        target_sr=target_sr,
        target_length=clip_duration
    )
    
    # If we have more ESC-50 samples than needed, create a subset
    if len(esc50_dataset) > num_background_samples:
        # Randomly select indices
        indices = random.sample(range(len(esc50_dataset)), num_background_samples)
        esc50_dataset = torch.utils.data.Subset(esc50_dataset, indices)
    
    # Combine the datasets
    combined_dataset = ConcatDataset([bird_dataset, esc50_dataset])
    
    print(f"Created combined dataset with {len(bird_dataset)} bird samples and {len(esc50_dataset)} background samples")
    
    return combined_dataset

if __name__==  "__main__":
    print("Testing BirdSoundDataset")
    print("------------------------")
    
    # Check if directory exists
    bird_dir = "bird_sound_dataset"
    if not os.path.exists(bird_dir):
        print(f"Creating directory {bird_dir}...")
        os.makedirs(bird_dir, exist_ok=True)
        print("Create some example folders with audio files to test!")
    else:
        print(f"Directory {bird_dir} found!")
        
        # List contents
        dirs = [d for d in os.listdir(bird_dir) if os.path.isdir(os.path.join(bird_dir, d))]
        print(f"Folders found: {dirs}")
        
        for d in dirs:
            files = [f for f in os.listdir(os.path.join(bird_dir, d)) 
                   if f.endswith(('.wav', '.mp3', '.ogg', '.flac'))]
            print(f"  {d}: {len(files)} audio files")
    
    test_dataset = BirdSoundDataset(root_dir=bird_dir, subset="testing", extract_calls=True)
    print(f"Dataset created with {len(test_dataset)} samples")
    
    # Test loading a sample
    if len(test_dataset) > 0:
        print("\nTesting sample loading:")
        try:
            sample, label = test_dataset[0]
            print(f"Sample loaded: shape={sample.shape}, label={label}")
            print("Test completed successfully!")
        except Exception as e:
            print(f"Error loading sample: {e}")
    else:
        print("Dataset is empty. Add some audio files to the folders before testing.")
    
    # Test ESC-50 functionality
    print("\nTesting ESC-50 dataset")
    print("----------------------")
    
    test_esc50 = input("Do you want to test ESC-50 download and integration? (y/n): ")
    if test_esc50.lower() == 'y':
        print("Downloading and extracting ESC-50 dataset...")
        esc50_dir = download_and_extract_esc50()
        
        print(f"Testing ESC-50 dataset from {esc50_dir}")
        try:
            esc50_dataset = ESC50Dataset(
                root_dir=esc50_dir,
                fold=[1],  # Just use fold 1 for testing
                exclude_bird_classes=True
            )
            print(f"ESC-50 dataset created with {len(esc50_dataset)} samples")
            
            if len(esc50_dataset) > 0:
                sample, label = esc50_dataset[0]
                print(f"ESC-50 sample loaded: shape={sample.shape}, category={esc50_dataset.categories[label]}")
                
                # Test combined dataset creation
                if len(dirs) > 0:  # Only if we have bird folders
                    print("\nTesting combined dataset creation...")
                    combined_dataset = create_combined_dataset(
                        bird_data_dir=bird_dir,
                        esc50_dir=esc50_dir,
                        allowed_bird_classes=dirs,
                        subset="training"
                    )
                    print(f"Combined dataset created with {len(combined_dataset)} total samples")
                else:
                    print("Can't test combined dataset without bird folders. Please add some bird species folders first.")
            else:
                print("No ESC-50 samples available")
        except Exception as e:
            print(f"Error testing ESC-50: {e}")
    else:
        print("Skipping ESC-50 testing")