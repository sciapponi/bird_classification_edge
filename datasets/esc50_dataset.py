"""
ESC-50 Dataset

This module provides the implementation of the ESC-50 (Environmental Sound Classification)
dataset for audio classification tasks.
"""

import os
import requests
import zipfile
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

def download_and_extract_esc50():
    """
    Downloads and extracts the ESC-50 dataset if it doesn't exist.
    
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
    
    # Create directory if needed
    os.makedirs(os.path.dirname(esc50_zip), exist_ok=True)
    
    # Download the dataset
    print("Downloading ESC-50 dataset...")
    response = requests.get(esc50_url, stream=True)
    with open(esc50_zip, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

    # Extract the dataset
    print("Extracting ESC-50 dataset...")
    with zipfile.ZipFile(esc50_zip, "r") as zip_ref:
        zip_ref.extractall(".")
    
    # Clean up
    os.remove(esc50_zip)
    
    print("ESC-50 dataset downloaded and extracted.")
    return esc50_dir

class ESC50Dataset(Dataset):
    """
    Environmental Sound Classification Dataset (ESC-50).
    Used to provide non-bird sound samples as negative examples or background noise.
    """
    def __init__(self, root_dir, fold=None, select_categories=None,
                 transform=None, target_sr=22050, target_length=3.0):
        """
        Initialize the ESC-50 dataset.
        
        Args:
            root_dir (str): Root directory of ESC-50 dataset
            fold (int or list, optional): Which fold(s) to use (1-5), None for all
            select_categories (list, optional): List of categories to include. If None, all are included.
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
        
        # Filter by selected categories if specified
        if select_categories:
            print(f"Filtering ESC-50 dataset for categories: {select_categories}")
            self.df = self.df[self.df['category'].isin(select_categories)]
            print(f"  {len(self.df)} samples remaining after category filtering.")
            if len(self.df) == 0:
                 print(f"WARNING: No samples found for the selected categories: {select_categories} in folds {fold}")
        
        # Create a mapping of category to index (using all original categories for consistency if needed elsewhere,
        # but labels returned will correspond to filtered data)
        all_original_categories = sorted(pd.read_csv(self.meta_file)['category'].unique())
        self.category_to_idx = {category: i for i, category in enumerate(all_original_categories)}
        
        # For the actual labels used by this dataset instance, map the filtered categories
        self.filtered_categories = sorted(self.df['category'].unique())
        self.filtered_category_to_idx = {category: i for i, category in enumerate(self.filtered_categories)}
        
        # Create file list and labels based on the filtered dataframe
        self.file_list = [os.path.join(self.audio_dir, filename) for filename in self.df['filename']]
        
        # Use the filtered category mapping for labels
        self.labels = [self.filtered_category_to_idx[category] for category in self.df['category']]
        
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.file_list)
    
    def __getitem__(self, idx):
        """
        Returns the audio sample and its label at the given index.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (waveform, label)
        """
        audio_path = self.file_list[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        waveform = self.load_audio(audio_path)
        
        # Apply transform if specified
        if self.transform:
            waveform = self.transform(waveform)
        
        return waveform, label
    
    def load_audio(self, audio_path):
        """
        Load and preprocess audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            torch.Tensor: Processed waveform
        """
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
                start = torch.randint(0, waveform.shape[1] - target_length_samples + 1, (1,)).item()
                waveform = waveform[:, start:start + target_length_samples]
            
            # Normalize audio
            if waveform.abs().max() > 0:
                waveform = waveform / waveform.abs().max()
            
            return waveform
        
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            # Return a silent clip as fallback
            return torch.zeros(1, int(self.target_sr * self.target_length)) 