import os
import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from datasets.dataset_factory import create_combined_dataset

class DistillationBirdSoundDataset:
    """
    Dataset wrapper that combines dataset_factory functionality with soft labels
    for knowledge distillation training.
    """
    
    def __init__(self, soft_labels_path, **dataset_kwargs):
        """
        Initialize distillation dataset.
        
        Args:
            soft_labels_path: Path to directory containing soft_labels.json
            **dataset_kwargs: Arguments for dataset_factory.create_combined_dataset
        """
        self.soft_labels_path = Path(soft_labels_path)
        self.soft_labels_data = {}
        self.soft_labels_metadata = {}
        
        # Load soft labels first
        self._load_soft_labels()
        
        # Create the underlying dataset using dataset_factory
        self.base_dataset = create_combined_dataset(**dataset_kwargs)
        
        print(f"DistillationDataset initialized:")
        print(f"  Base dataset samples: {len(self.base_dataset)}")
        print(f"  Soft labels available: {len(self.soft_labels_data)}")
        print(f"  Soft labels classes: {self.soft_labels_metadata.get('num_classes', 'Unknown')}")
    
    def _load_soft_labels(self):
        """Load soft labels from JSON file"""
        soft_labels_file = self.soft_labels_path / "soft_labels.json"
        metadata_file = self.soft_labels_path / "soft_labels_metadata.json"
        
        if not soft_labels_file.exists():
            raise FileNotFoundError(f"Soft labels file not found: {soft_labels_file}")
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        # Load soft labels
        with open(soft_labels_file, 'r') as f:
            self.soft_labels_data = json.load(f)
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            self.soft_labels_metadata = json.load(f)
        
        print(f"Loaded soft labels for {len(self.soft_labels_data)} files")
        print(f"Teacher confidence threshold: {self.soft_labels_metadata.get('confidence_threshold', 'Unknown')}")
    
    def _get_soft_label(self, file_path):
        """
        Get soft label for a given file path.
        
        Args:
            file_path: Absolute path to audio file
            
        Returns:
            numpy array of soft labels, or zeros if not found
        """
        # Convert absolute path to relative path (relative to dataset root)
        try:
            # Get relative path from the dataset
            if hasattr(self.base_dataset, 'root_dir'):
                rel_path = str(Path(file_path).relative_to(self.base_dataset.root_dir))
            else:
                # Fallback: extract relative path from file_path
                # Assuming structure: .../bird_sound_dataset/Species/file.wav
                parts = Path(file_path).parts
                if 'bird_sound_dataset' in parts:
                    dataset_idx = parts.index('bird_sound_dataset')
                    rel_path = str(Path(*parts[dataset_idx + 1:]))
                elif 'augmented_dataset' in parts:
                    # Handle non-bird files from augmented_dataset
                    dataset_idx = parts.index('augmented_dataset')
                    rel_path = str(Path(*parts[dataset_idx + 1:]))
                else:
                    # Last resort: just species/filename
                    rel_path = str(Path(*parts[-2:]))
            
            # Get soft labels
            if rel_path in self.soft_labels_data:
                soft_labels = np.array(self.soft_labels_data[rel_path], dtype=np.float32)
                
                # Ensure we have exactly 5 classes (4 birds + non-bird)
                if len(soft_labels) != 5:
                    # If somehow we have different number of classes, pad or truncate
                    if len(soft_labels) < 5:
                        # Pad with zeros for missing classes
                        padded = np.zeros(5, dtype=np.float32)
                        padded[:len(soft_labels)] = soft_labels
                        soft_labels = padded
                    else:
                        # Truncate if more than 5 classes
                        soft_labels = soft_labels[:5]
                        
            else:
                # Not found, create default soft labels based on file path
                soft_labels = np.zeros(5, dtype=np.float32)
                
                # Check if this is a non-bird file
                if 'no_birds' in rel_path or 'empty_' in rel_path:
                    # Non-bird file: high confidence for non-bird class (index 4)
                    soft_labels[4] = 1.0
                else:
                    # Bird file: distribute equally among bird classes (indices 0-3)
                    soft_labels[:4] = 0.25
                
                # Debug: print missing files occasionally
                if np.random.random() < 0.01:  # Print only 1% of missing files
                    print(f"Warning: No soft labels found for {rel_path}")
                    print(f"Using default: {'non-bird' if soft_labels[4] > 0.5 else 'bird'} distribution")
            
            return soft_labels
            
        except Exception as e:
            # Fallback based on file path
            print(f"Error getting soft labels for {file_path}: {e}")
            soft_labels = np.zeros(5, dtype=np.float32)
            
            # Check if this looks like a non-bird file
            if 'no_birds' in str(file_path) or 'empty_' in str(file_path):
                soft_labels[4] = 1.0  # Non-bird class
            else:
                soft_labels[:4] = 0.25  # Equal distribution for bird classes
                
            return soft_labels
    
    def __getitem__(self, idx):
        """
        Get item with both hard and soft labels.
        
        Returns:
            tuple: (audio_tensor, hard_label, soft_label_tensor)
        """
        # Get base item (audio, hard_label)
        audio, hard_label = self.base_dataset[idx]
        
        # Get corresponding file path - handle ConcatDataset
        file_path = self._get_file_path(idx)
        
        # Get soft labels
        soft_labels = self._get_soft_label(file_path)
        soft_label_tensor = torch.from_numpy(soft_labels)
        
        return audio, hard_label, soft_label_tensor
    
    def _get_file_path(self, idx):
        """Get file path for a given index, handling ConcatDataset"""
        if hasattr(self.base_dataset, 'file_list'):
            # Simple case: direct BirdSoundDataset
            return self.base_dataset.file_list[idx]
        elif hasattr(self.base_dataset, 'datasets'):
            # ConcatDataset case: find which subdataset contains this index
            dataset_idx = 0
            current_idx = idx
            
            for dataset in self.base_dataset.datasets:
                if current_idx < len(dataset):
                    # Found the right dataset
                    if hasattr(dataset, 'file_list'):
                        return dataset.file_list[current_idx]
                    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'file_list'):
                        # Subset or wrapper dataset
                        # Get the actual file index if it's a Subset
                        if hasattr(dataset, 'indices'):
                            actual_idx = dataset.indices[current_idx]
                            return dataset.dataset.file_list[actual_idx]
                        else:
                            return dataset.dataset.file_list[current_idx]
                    else:
                        # Unknown dataset type, generate a dummy path
                        return f"unknown_dataset_{dataset_idx}/sample_{current_idx}.wav"
                else:
                    # Move to next dataset
                    current_idx -= len(dataset)
                    dataset_idx += 1
            
            # Fallback if not found
            return f"unknown/sample_{idx}.wav"
        else:
            # Unknown dataset structure
            return f"unknown/sample_{idx}.wav"
    
    def get_soft_labels_info(self):
        """Get information about soft labels"""
        return {
            'num_classes': self.soft_labels_metadata.get('num_classes'),
            'target_species': self.soft_labels_metadata.get('target_species'),
            'confidence_threshold': self.soft_labels_metadata.get('confidence_threshold'),
            'total_files_with_soft_labels': len(self.soft_labels_data),
            'files_processed': self.soft_labels_metadata.get('total_files_processed')
        }
    
    def get_classes(self):
        """Get list of class names (compatible method)"""
        # For ConcatDataset (from dataset_factory), we know the structure:
        # 4 bird classes + 1 non-bird class
        if hasattr(self.base_dataset, 'allowed_classes'):
            return self.base_dataset.allowed_classes
        else:
            # Fallback: return the expected 5 classes based on soft labels metadata
            target_species = self.soft_labels_metadata.get('target_species', [])
            if target_species:
                return target_species
            else:
                # Final fallback: hardcoded for our case
                return ['Poecile_montanus', 'Certhia_familiaris', 'Apus_apus', 'Bubo_bubo', 'non-bird']

    def __len__(self):
        """Return length of dataset"""
        return len(self.base_dataset)

def create_distillation_dataloader(config, soft_labels_path, split='train'):
    """
    Create a DataLoader for distillation training.
    
    Args:
        config: Dataset configuration
        soft_labels_path: Path to soft labels directory
        split: 'train', 'val', or 'test'
        
    Returns:
        torch.utils.data.DataLoader, dataset
    """
    from torch.utils.data import DataLoader
    
    # Map split names
    if split == 'train':
        subset = 'training'
    elif split == 'val':
        subset = 'validation'
    else:
        subset = 'testing'
    
    # Create distillation dataset
    dataset = DistillationBirdSoundDataset(
        soft_labels_path=soft_labels_path,
        # Parameters for dataset_factory.create_combined_dataset
        bird_data_dir=config.main_data_dir,
        esc50_dir=config.get('esc50_dir', 'ESC-50-master'),
        allowed_bird_classes=config.allowed_bird_classes,
        target_sr=config.sample_rate,
        clip_duration=config.clip_duration,
        subset=subset,
        validation_split=config.get('val_split', 0.15),
        test_split=config.get('test_split', 0.15),
        split_seed=config.get('seed', 42),
        use_augmentation=config.augmentation.enabled if split == 'train' else False,
        # Non-bird parameters for dataset_factory
        load_pregenerated_no_birds=config.get('load_pregenerated_no_birds', False),
        pregenerated_no_birds_dir=config.get('pregenerated_no_birds_dir', 'augmented_dataset/no_birds'),
        num_no_bird_samples=config.get('num_no_bird_samples', 100)
    )
    
    # Create dataloader
    batch_size = config.get('batch_size', 8)
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get('num_workers', 0),
        pin_memory=True
    )
    
    return dataloader, dataset

def test_distillation_dataset():
    """Test function for distillation dataset"""
    from omegaconf import DictConfig
    
    # Mock configuration
    config = DictConfig({
        'main_data_dir': 'bird_sound_dataset',
        'allowed_bird_classes': None,
        'sample_rate': 32000,
        'clip_duration': 3.0,
        'extract_calls': False,
        'lowcut': 150.0,
        'highcut': 16000.0,
        'augmentation': {'enabled': False},
        'batch_size': 4
    })
    
    try:
        # Test dataset creation
        dataset = DistillationBirdSoundDataset(
            soft_labels_path='soft_labels',
            root_dir=config.main_data_dir,
            allowed_classes=config.allowed_bird_classes,
            sr=config.sample_rate,
            clip_duration=config.clip_duration,
            subset='training'
        )
        
        # Test data loading
        audio, hard_label, soft_labels = dataset[0]
        print(f"Audio shape: {audio.shape}")
        print(f"Hard label: {hard_label}")
        print(f"Soft labels shape: {soft_labels.shape}")
        print(f"Soft labels sum: {soft_labels.sum():.4f}")
        print("Test passed!")
        
    except FileNotFoundError as e:
        print(f"Test skipped (missing files): {e}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_distillation_dataset() 