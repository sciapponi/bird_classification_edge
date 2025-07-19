#!/usr/bin/env python3
"""
Hybrid dataset that can load either preprocessed audio files or original audio files
based on configuration, maintaining compatibility with the distillation training pipeline.
"""

import os
import sys
import json
import numpy as np
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.audio_utils import extract_call_segments

logger = logging.getLogger(__name__)

class HybridBirdDataset(Dataset):
    """
    Hybrid dataset that can load either:
    1. Preprocessed audio files (WAV/MP3 with preprocessing already applied)
    2. Original audio files (with on-the-fly preprocessing)
    3. NPY/NPZ files (numpy arrays)
    
    The choice is controlled by the 'use_preprocessed' flag in the config.
    """
    
    def __init__(self, root_dir, split='train', use_preprocessed=False, 
                 preprocessed_dir=None, audio_config=None, soft_labels_path=None):
        """
        Args:
            root_dir: Path to original audio dataset
            split: 'train', 'val', or 'test'
            use_preprocessed: If True, load from preprocessed_dir. If False, use original files
            preprocessed_dir: Path to preprocessed files (if use_preprocessed=True)
            audio_config: Audio processing configuration
            soft_labels_path: Path to soft labels JSON file
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.use_preprocessed = use_preprocessed
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        
        # Default audio config (matches training pipeline)
        self.audio_config = audio_config or {
            'sample_rate': 32000,
            'clip_duration': 3.0,
            'lowcut': 150.0,
            'highcut': 16000.0,
            'extract_calls': True
        }
        
        # Load class mapping
        self.class_to_idx = self._create_class_mapping()
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Find files
        if self.use_preprocessed and self.preprocessed_dir:
            self.files = self._find_preprocessed_files()
            logger.info(f"Loading from preprocessed files: {len(self.files)} files found")
        else:
            self.files = self._find_original_files()
            logger.info(f"Loading from original files: {len(self.files)} files found")
        
        # Load soft labels if provided
        self.soft_labels = {}
        if soft_labels_path:
            self._load_soft_labels(soft_labels_path)
            
    def _create_class_mapping(self):
        """Create mapping from class names to indices."""
        if self.use_preprocessed and self.preprocessed_dir and self.preprocessed_dir.exists():
            # Get classes from preprocessed directory structure
            class_dirs = [d for d in self.preprocessed_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
        else:
            # Get classes from original directory structure
            class_dirs = [d for d in self.root_dir.iterdir() 
                         if d.is_dir() and not d.name.startswith('.')]
        
        classes = sorted([d.name for d in class_dirs])
        return {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    def _find_original_files(self):
        """Find original audio files."""
        files = []
        audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg'}
        
        for class_name in self.class_to_idx.keys():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
                
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in audio_extensions:
                    files.append({
                        'path': file_path,
                        'class': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'type': 'original'
                    })
        
        return files
    
    def _find_preprocessed_files(self):
        """Find preprocessed files."""
        files = []
        # Support both audio files and numpy arrays
        preprocessed_extensions = {'.wav', '.mp3', '.npy', '.npz'}
        
        for class_name in self.class_to_idx.keys():
            class_dir = self.preprocessed_dir / class_name
            if not class_dir.exists():
                continue
                
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in preprocessed_extensions:
                    files.append({
                        'path': file_path,
                        'class': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'type': 'preprocessed'
                    })
        
        return files
    
    def _load_soft_labels(self, soft_labels_path):
        """Load soft labels from JSON file."""
        try:
            with open(soft_labels_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, dict):
                # Format: {"filename": [soft_labels]}
                for filename, soft_labels in data.items():
                    self.soft_labels[filename] = torch.tensor(soft_labels, dtype=torch.float32)
                    name_without_ext = Path(filename).stem
                    self.soft_labels[name_without_ext] = torch.tensor(soft_labels, dtype=torch.float32)
            elif isinstance(data, list):
                # Format: [{"filename": "...", "soft_labels": [...]}]
                for item in data:
                    filename = Path(item['filename']).name
                    self.soft_labels[filename] = torch.tensor(item['soft_labels'], dtype=torch.float32)
                    name_without_ext = Path(filename).stem
                    self.soft_labels[name_without_ext] = torch.tensor(item['soft_labels'], dtype=torch.float32)
                
            logger.info(f"Loaded {len(data)} soft labels")
        except Exception as e:
            logger.warning(f"Could not load soft labels: {e}")
            self.soft_labels = {}
    
    def _load_original_audio(self, file_path):
        """Load and preprocess original audio file."""
        try:
            # Use extract_call_segments which matches the training pipeline exactly
            call_intervals, segments, _, _, _ = extract_call_segments(
                str(file_path),
                clip_duration=self.audio_config['clip_duration'],
                sr=self.audio_config['sample_rate'],
                lowcut=self.audio_config['lowcut'],
                highcut=self.audio_config['highcut']
            )
            
            # Use the first segment if available, otherwise return zeros
            if segments and len(segments) > 0:
                audio_data = segments[0]
            else:
                # No calls detected, create silence
                target_length = int(self.audio_config['sample_rate'] * self.audio_config['clip_duration'])
                audio_data = np.zeros(target_length, dtype=np.float32)
            
            if audio_data is None:
                raise ValueError("Failed to extract audio segments")
                
            return torch.from_numpy(audio_data).float()
            
        except Exception as e:
            logger.warning(f"Error loading original audio {file_path}: {e}")
            # Return silence as fallback
            target_length = int(self.audio_config['sample_rate'] * self.audio_config['clip_duration'])
            return torch.zeros(target_length, dtype=torch.float32)
    
    def _load_preprocessed_audio(self, file_path):
        """Load preprocessed audio file."""
        try:
            if file_path.suffix == '.npz':
                # Load NPZ file
                data = np.load(file_path, allow_pickle=True)
                audio_data = data['audio']
                return torch.from_numpy(audio_data).float()
                
            elif file_path.suffix == '.npy':
                # Load NPY file
                audio_data = np.load(file_path)
                return torch.from_numpy(audio_data).float()
                
            elif file_path.suffix in ['.wav', '.mp3']:
                # Load audio file (already preprocessed)
                audio_data, sr = torchaudio.load(file_path)
                
                # Ensure correct sample rate
                if sr != self.audio_config['sample_rate']:
                    resampler = torchaudio.transforms.Resample(sr, self.audio_config['sample_rate'])
                    audio_data = resampler(audio_data)
                
                # Convert to mono if needed
                if audio_data.shape[0] > 1:
                    audio_data = audio_data.mean(dim=0, keepdim=False)
                else:
                    audio_data = audio_data.squeeze(0)
                
                return audio_data.float()
            
        except Exception as e:
            logger.warning(f"Error loading preprocessed audio {file_path}: {e}")
            # Return silence as fallback
            target_length = int(self.audio_config['sample_rate'] * self.audio_config['clip_duration'])
            return torch.zeros(target_length, dtype=torch.float32)
    
    def _get_soft_labels(self, file_path):
        """Get soft labels for a file."""
        # Try multiple filename variants
        filename = file_path.name
        filename_stem = file_path.stem
        
        # Remove common preprocessing suffixes to match original names
        clean_stem = filename_stem
        for suffix in ['_preprocessed', '_processed', '_3s', '_32k']:
            clean_stem = clean_stem.replace(suffix, '')
        
        # Try to find soft labels
        for key in [filename, filename_stem, clean_stem]:
            if key in self.soft_labels:
                return self.soft_labels[key]
        
        # Return uniform distribution as fallback
        num_classes = self.num_classes
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_info = self.files[idx]
        file_path = file_info['path']
        class_idx = file_info['class_idx']
        
        # Load audio based on type
        if file_info['type'] == 'original':
            audio = self._load_original_audio(file_path)
        else:  # preprocessed
            audio = self._load_preprocessed_audio(file_path)
        
        # Get hard and soft labels
        hard_label = torch.tensor(class_idx, dtype=torch.long)
        soft_label = self._get_soft_labels(file_path)
        
        return audio, hard_label, soft_label
    
    def get_classes(self):
        """Return list of class names."""
        return [self.idx_to_class[i] for i in range(len(self.class_to_idx))]
    
    def get_soft_labels_info(self):
        """Return information about soft labels."""
        return {
            'num_classes': len(self.class_to_idx),
            'soft_labels_available': len(self.soft_labels) > 0,
            'total_soft_labels': len(self.soft_labels)
        }

def create_hybrid_dataloader(config, soft_labels_path, split='train'):
    """
    Create a hybrid dataloader that can use either preprocessed or original files.
    
    Args:
        config: Dataset configuration with 'use_preprocessed' flag
        soft_labels_path: Path to soft labels
        split: Dataset split
    
    Returns:
        DataLoader and Dataset
    """
    from torch.utils.data import DataLoader
    
    # Extract configuration
    use_preprocessed = config.get('use_preprocessed', False)
    original_dataset_path = config.get('dataset_path', 'bird_sound_dataset')
    preprocessed_dataset_path = config.get('preprocessed_dataset_path', 'preprocessed_dataset')
    
    # Audio processing config
    audio_config = {
        'sample_rate': config.get('sample_rate', 32000),
        'clip_duration': config.get('clip_duration', 3.0),
        'lowcut': config.get('lowcut', 150.0),
        'highcut': config.get('highcut', 16000.0),
        'extract_calls': config.get('extract_calls', True)
    }
    
    # Create dataset
    dataset = HybridBirdDataset(
        root_dir=original_dataset_path,
        split=split,
        use_preprocessed=use_preprocessed,
        preprocessed_dir=preprocessed_dataset_path if use_preprocessed else None,
        audio_config=audio_config,
        soft_labels_path=soft_labels_path
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=(split == 'train'),
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return dataloader, dataset 