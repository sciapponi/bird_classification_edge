#!/usr/bin/env python3
"""
Training script for knowledge distillation using BirdNET as teacher.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import json
import time
import gc

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from distillation.losses.distillation_loss import DistillationLoss, AdaptiveDistillationLoss
from distillation.losses.focal_loss import FocalLoss, FocalDistillationLoss, AdaptiveFocalDistillationLoss
from distillation.datasets.distillation_dataset import create_distillation_dataloader
from distillation.datasets.hybrid_dataset import create_hybrid_dataloader
from models import Improved_Phi_GRU_ATT
# from utils.metrics import calculate_metrics, save_confusion_matrix
# from utils.training_utils import EarlyStopping, save_checkpoint, load_checkpoint

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple implementations for testing
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

class PreprocessedDistillationDataset:
    """
    Wrapper dataset that combines preprocessed files with soft labels for distillation.
    """
    
    def __init__(self, preprocessed_dataset, soft_labels_path):
        """
        Initialize preprocessed distillation dataset.
        
        Args:
            preprocessed_dataset: Instance of PreprocessedBirdDataset
            soft_labels_path: Path to directory containing soft_labels.json
        """
        self.preprocessed_dataset = preprocessed_dataset
        self.soft_labels_path = Path(soft_labels_path)
        self.soft_labels_data = {}
        self.soft_labels_metadata = {}
        
        # Load soft labels
        self._load_soft_labels()
        
        logger.info(f"PreprocessedDistillationDataset initialized:")
        logger.info(f"  Preprocessed dataset samples: {len(self.preprocessed_dataset)}")
        logger.info(f"  Soft labels available: {len(self.soft_labels_data)}")
        logger.info(f"  Soft labels classes: {self.soft_labels_metadata.get('num_classes', 'Unknown')}")
    
    def _load_soft_labels(self):
        """Load soft labels from JSON file"""
        soft_labels_file = self.soft_labels_path / "soft_labels.json"
        metadata_file = self.soft_labels_path / "soft_labels_metadata.json"
        
        if not soft_labels_file.exists():
            raise FileNotFoundError(f"Soft labels file not found: {soft_labels_file}")
        
        # Load soft labels data
        with open(soft_labels_file, 'r') as f:
            self.soft_labels_data = json.load(f)
        
        # Load metadata if available
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.soft_labels_metadata = json.load(f)
        
        logger.info(f"Loaded soft labels for {len(self.soft_labels_data)} files")
    
    def __len__(self):
        return len(self.preprocessed_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with audio, hard label, and soft label.
        
        Returns:
            tuple: (audio_tensor, hard_label, soft_label_tensor)
        """
        # Get preprocessed audio and metadata from combined dataset
        if self.preprocessed_dataset.return_metadata:
            audio, class_idx, metadata = self.preprocessed_dataset[idx]
            original_filename = metadata.get('original_filename', '')
        else:
            audio, class_idx = self.preprocessed_dataset[idx]
            # Try to get filename from dataset internal data
            sample_data = self.preprocessed_dataset.samples[idx]
            original_filename = sample_data.get('original_filename', sample_data.get('filename', ''))
        
        # Use class_idx directly from the combined dataset (handles both bird and no_bird)
        hard_label = class_idx
        
        # Get soft label
        soft_label = self._get_soft_label_for_file(original_filename)
        
        # Ensure consistent audio tensor shape (1D)
        if audio.dim() > 1:
            audio = audio.squeeze()  # Remove extra dimensions
        audio = audio.view(-1)  # Force to 1D
        
        return audio, hard_label, soft_label
    
    def _get_soft_label_for_file(self, filename):
        """
        Get soft label for a given filename.
        
        Args:
            filename: Original filename
            
        Returns:
            torch.Tensor: Soft label probabilities
        """
        # Remove extension and path to match soft labels keys
        base_filename = Path(filename).stem
        
        # Try to find soft label by exact match first
        if base_filename in self.soft_labels_data:
            soft_label = self.soft_labels_data[base_filename]
        else:
            # Try partial matching (filename might contain additional info)
            matching_keys = [key for key in self.soft_labels_data.keys() if base_filename in key or key in base_filename]
            if matching_keys:
                # Use first match
                soft_label = self.soft_labels_data[matching_keys[0]]
                logger.debug(f"Soft label found using partial match: {base_filename} -> {matching_keys[0]}")
            else:
                # FIXED: Use consistent number of classes
                # Get the expected number of classes from metadata, with fallback
                expected_num_classes = self.soft_labels_metadata.get('num_classes')
                if not expected_num_classes:
                    # If metadata is missing, check the size of existing soft labels
                    if self.soft_labels_data:
                        # Get size from any existing soft label
                        sample_key = next(iter(self.soft_labels_data))
                        expected_num_classes = len(self.soft_labels_data[sample_key])
                    else:
                        # Last resort: use dataset classes + 1 for non-bird
                        dataset_classes = self.preprocessed_dataset.class_to_idx if hasattr(self.preprocessed_dataset, 'class_to_idx') else {}
                        expected_num_classes = len(dataset_classes) + 1  # +1 for non-bird
                
                # Create uniform distribution with correct size
                soft_label = [1.0 / expected_num_classes] * expected_num_classes
                logger.warning(f"No soft label found for {base_filename}, using uniform distribution with {expected_num_classes} classes")
        
        return torch.tensor(soft_label, dtype=torch.float32)
    
    def get_soft_labels_info(self):
        """Get information about soft labels"""
        return {
            'total_files_with_soft_labels': len(self.soft_labels_data),
            'num_classes': self.soft_labels_metadata.get('num_classes', len(self.get_classes())),
            'metadata': self.soft_labels_metadata
        }
    
    def get_classes(self):
        """Get list of class names from the soft labels metadata."""
        # The source of truth for classes should be the teacher's labels.
        target_species = self.soft_labels_metadata.get('target_species')
        num_classes = self.soft_labels_metadata.get('num_classes', 0)
        
        if not target_species or len(target_species) != num_classes:
            print("Warning: 'target_species' in metadata doesn't match 'num_classes'. Falling back to base dataset classes.")
            
            # For preprocessed datasets, use idx_to_class mapping
            if hasattr(self.preprocessed_dataset, 'idx_to_class'):
                return [self.preprocessed_dataset.idx_to_class[i] for i in range(len(self.preprocessed_dataset.idx_to_class))]
            # For other datasets, try class_to_idx
            elif hasattr(self.preprocessed_dataset, 'class_to_idx'):
                return list(self.preprocessed_dataset.class_to_idx.keys())
            # Fallback to generic names
            else:
                return [f"Class_{i}" for i in range(num_classes)]
            
        return target_species

def save_checkpoint(model, optimizer, epoch, val_loss, val_acc, filepath='checkpoint.pt'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, filepath)

def load_checkpoint(filepath):
    return torch.load(filepath)

def save_confusion_matrix(y_true, y_pred, class_names, png_path, csv_path):
    """Saves the confusion matrix as a PNG image and a CSV file."""
    logger.info(f"Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    
    # Save CSV
    try:
        cm_df.to_csv(csv_path)
        logger.info(f"Confusion matrix CSV saved to {csv_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix CSV: {e}")

    # Save PNG with better formatting for many classes
    num_classes = len(class_names)
    
    # Adaptive figure size based on number of classes
    if num_classes > 50:
        figsize = (20, 18)  # Large figure for 70+ classes
        annot_fontsize = 6  # Small font for annotations
        label_fontsize = 8  # Small font for labels
        title_fontsize = 14
    elif num_classes > 20:
        figsize = (16, 14)
        annot_fontsize = 8
        label_fontsize = 10
        title_fontsize = 16
    else:
        figsize = (12, 10)
        annot_fontsize = 10
        label_fontsize = 12
        title_fontsize = 18
    
    plt.figure(figsize=figsize)
    
    # Create heatmap with conditional annotation
    if num_classes > 50:
        # For large matrices, only show annotations for non-zero values to reduce clutter
        annot_array = np.where(cm > 0, cm, "")
        sns.heatmap(cm_df, annot=annot_array, fmt='', cmap='Blues', 
                   cbar_kws={'shrink': 0.8}, annot_kws={'size': annot_fontsize})
    else:
        # For smaller matrices, show all annotations
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
                   cbar_kws={'shrink': 0.8}, annot_kws={'size': annot_fontsize})
    
    plt.title('Confusion Matrix', fontsize=title_fontsize, pad=20)
    plt.ylabel('Actual', fontsize=label_fontsize)
    plt.xlabel('Predicted', fontsize=label_fontsize)
    
    # Rotate labels for better readability with many classes
    if num_classes > 20:
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
    else:
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    try:
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix PNG saved to {png_path}")
        logger.info(f"Matrix size: {num_classes}x{num_classes} classes")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix PNG: {e}")
    finally:
        plt.close()

class DistillationTrainer:
    def __init__(self, config, soft_labels_path, output_dir):
        self.config = config
        self.soft_labels_path = soft_labels_path
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.test_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.hard_losses = []
        self.soft_losses = []
        self.learning_rates = []  # Track learning rates
        
        # Filter parameters history (for combined_log_linear)
        self.breakpoint_history = []
        self.transition_width_history = []
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Outputs will be saved to: {self.output_dir}")
    
    def setup_data(self):
        """Setup data loaders for distillation training"""
        logger.info("Setting up data loaders with soft labels...")
        
        # Check if we should use preprocessed files (no runtime processing)
        use_preprocessed_files = not self.config.dataset.get('process', True)  # Default: process=True (do preprocessing)
        
        if use_preprocessed_files:
            logger.info("Using Preprocessed Files (no runtime preprocessing)")
            
            # Create preprocessed dataloaders with soft labels
            self.train_loader, train_dataset = self._create_preprocessed_distillation_dataloader(split='train')
            self.val_loader, val_dataset = self._create_preprocessed_distillation_dataloader(split='validation')  
            self.test_loader, test_dataset = self._create_preprocessed_distillation_dataloader(split='test')
            
        else:
            # Check if we should use hybrid dataset (supports both preprocessed and original files)
            use_hybrid_dataset = self.config.dataset.get('use_hybrid', False)
            
            if use_hybrid_dataset:
                logger.info("Using Hybrid Dataset (supports preprocessed and original files)")
                
                # Create train loader
                self.train_loader, train_dataset = create_hybrid_dataloader(
                    self.config.dataset, self.soft_labels_path, split='train'
                )
                
                # Create validation loader
                self.val_loader, val_dataset = create_hybrid_dataloader(
                    self.config.dataset, self.soft_labels_path, split='val'
                )
                
                # Create test loader
                self.test_loader, test_dataset = create_hybrid_dataloader(
                    self.config.dataset, self.soft_labels_path, split='test'
                )
                
            else:
                logger.info("Using Original Distillation Dataset (with runtime preprocessing)")
                
                # Create train loader - pass training config for batch_size
                train_config = self.config.dataset.copy()
                if hasattr(self.config, 'training') and 'batch_size' in self.config.training:
                    # Temporarily disable struct mode to allow adding batch_size
                    with open_dict(train_config):
                        train_config['batch_size'] = self.config.training.batch_size
                    
                self.train_loader, train_dataset = create_distillation_dataloader(
                    train_config, self.soft_labels_path, split='train'
                )
                
                # Create validation loader - pass training config for batch_size
                val_config = self.config.dataset.copy()
                if hasattr(self.config, 'training') and 'batch_size' in self.config.training:
                    with open_dict(val_config):
                        val_config['batch_size'] = self.config.training.batch_size
                
                self.val_loader, val_dataset = create_distillation_dataloader(
                    val_config, self.soft_labels_path, split='val'
                )
                
                # Create test loader - pass training config for batch_size
                test_config = self.config.dataset.copy()
                if hasattr(self.config, 'training') and 'batch_size' in self.config.training:
                    with open_dict(test_config):
                        test_config['batch_size'] = self.config.training.batch_size
                
                self.test_loader, test_dataset = create_distillation_dataloader(
                    test_config, self.soft_labels_path, split='test'
                )
        
        # Log dataset info
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Get soft labels info and extract number of classes
        soft_info = train_dataset.get_soft_labels_info()
        
        # FIXED: Use dataset as source of truth, not soft labels
        # Get actual classes from the dataset
        dataset_classes = train_dataset.get_classes() if hasattr(train_dataset, 'get_classes') else []
        dataset_num_classes = len(dataset_classes)
        
        # Check if dataset already includes non-bird class
        dataset_already_has_non_bird = 'non-bird' in dataset_classes or 'no_bird' in dataset_classes
        include_non_bird = self.config.dataset.get('include_non_bird_class', True)  # Default: include
        
        if dataset_already_has_non_bird:
            # Dataset already includes non-bird class (like our CombinedPreprocessedDataset)
            self.num_classes = dataset_num_classes
            self.class_names = dataset_classes
            logger.info(f"Dataset already includes non-bird class: {self.num_classes} total classes")
        else:
            # Check if we need to add non-bird class (explicitly configured)
            if include_non_bird:
                self.num_classes = dataset_num_classes + 1  # Add non-bird class
                self.class_names = dataset_classes + ['non-bird']
                logger.info(f"Including non-bird class: {dataset_num_classes} bird classes + 1 non-bird = {self.num_classes} total")
            else:
                self.num_classes = dataset_num_classes
                self.class_names = dataset_classes
                logger.info(f"Using only bird classes: {self.num_classes} total")
        
        # Verify soft labels compatibility
        soft_num_classes = soft_info.get('num_classes', 0)
        if soft_num_classes != self.num_classes:
            logger.warning(f"Soft labels have {soft_num_classes} classes, but dataset configured for {self.num_classes} classes")
            logger.warning("This may cause issues during training. Consider regenerating soft labels or adjusting configuration.")
        
        logger.info(f"Soft labels info: {soft_info}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class names: {self.class_names}")
        logger.info(f"Final class verification: model will use {self.num_classes} classes")
        logger.info(f"Source of truth: Dataset ({dataset_num_classes} classes) + config (include_non_bird={include_non_bird})")
        
        # Log dataset type for clarity
        if use_preprocessed_files:
            logger.info("Using preprocessed files: No runtime preprocessing will be performed")
        elif use_hybrid_dataset:
            use_preprocessed = self.config.dataset.get('use_preprocessed', False)
            logger.info(f"Hybrid dataset mode: {'Preprocessed files' if use_preprocessed else 'Original files'}")
        else:
            logger.info("Original dataset mode: Runtime preprocessing will be performed")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_preprocessed_distillation_dataloader(self, split='train'):
        """
        Create dataloader for preprocessed files with soft labels for distillation.
        Automatically combines bird files (bird_sound_dataset_processed) and no_bird files (augmented_dataset/no_birds).
        
        Args:
            split: 'train', 'validation', or 'test'
            
        Returns:
            tuple: (dataloader, dataset)
        """
        import sys
        import os
        import random
        import torch
        import numpy as np
        from pathlib import Path
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
        from datasets.preprocessed_dataset import PreprocessedBirdDataset
        from torch.utils.data import DataLoader, ConcatDataset
        
        # Map split names
        subset_mapping = {
            'train': 'training',
            'validation': 'validation', 
            'test': 'test'
        }
        subset = subset_mapping.get(split, split)
        
        # Get configuration
        dataset_config = self.config.dataset
        
        # 1. Create preprocessed bird dataset (classes 0-69)
        bird_dataset = PreprocessedBirdDataset(
            root_dir=dataset_config.get('preprocessed_root_dir', dataset_config.get('root_dir')),
            allowed_classes=dataset_config.get('species_list', None),
            subset=subset,
            validation_split=dataset_config.get('validation_split', 0.15),
            test_split=dataset_config.get('test_split', 0.15),
            split_seed=dataset_config.get('split_seed', 42),
            transform=None,  # No transforms for preprocessed files
            return_metadata=True
        )
        
        logger.info(f"Loaded bird dataset: {len(bird_dataset)} samples, {len(bird_dataset.class_to_idx)} classes")
        
        # 2. Create no_bird dataset from augmented_dataset/no_birds (class 70)
        no_birds_dir = Path("augmented_dataset/no_birds")
        no_bird_files = []
        
        if no_birds_dir.exists():
            # Get all WAV files from no_birds directory
            all_no_bird_files = [f for f in no_birds_dir.iterdir() if f.suffix.lower() == '.wav']
            
            # Split no_bird files using same logic as bird dataset
            np.random.seed(dataset_config.get('split_seed', 42))
            indices = np.arange(len(all_no_bird_files))
            np.random.shuffle(indices)
            
            # Calculate split sizes
            validation_split = dataset_config.get('validation_split', 0.15)
            test_split = dataset_config.get('test_split', 0.15)
            
            n_files = len(all_no_bird_files)
            test_size = int(n_files * test_split)
            val_size = int(n_files * validation_split)
            train_size = n_files - test_size - val_size
            
            # Split indices
            if subset == "training":
                selected_indices = indices[:train_size]
            elif subset == "validation":
                selected_indices = indices[train_size:train_size + val_size]
            elif subset == "test":
                selected_indices = indices[train_size + val_size:]
            else:
                selected_indices = indices  # fallback
            
            # Get selected files
            no_bird_files = [all_no_bird_files[i] for i in selected_indices]
            
            logger.info(f"Loaded no_bird files: {len(no_bird_files)} samples for {subset}")
        else:
            logger.warning(f"No_birds directory not found: {no_birds_dir}")
        
        # 3. Create combined dataset class
        class CombinedPreprocessedDataset:
            def __init__(self, bird_dataset, no_bird_files):
                self.bird_dataset = bird_dataset
                self.no_bird_files = no_bird_files
                self.no_bird_class_idx = 70  # Fixed class index for no_bird
                
                # Create combined class mapping (70 bird classes + 1 no_bird class)
                self.class_to_idx = bird_dataset.class_to_idx.copy()
                self.class_to_idx['no_bird'] = self.no_bird_class_idx
                
                self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
                
                # Copy attributes from bird dataset for compatibility
                self.return_metadata = bird_dataset.return_metadata
                
            def __len__(self):
                return len(self.bird_dataset) + len(self.no_bird_files)
            
            def __getitem__(self, idx):
                if idx < len(self.bird_dataset):
                    # Bird sample
                    if self.bird_dataset.return_metadata:
                        audio, class_idx, metadata = self.bird_dataset[idx]
                        return audio, class_idx, metadata
                    else:
                        audio, class_idx = self.bird_dataset[idx]
                        return audio, class_idx
                else:
                    # No_bird sample
                    no_bird_idx = idx - len(self.bird_dataset)
                    no_bird_file = self.no_bird_files[no_bird_idx]
                    
                    # Load no_bird audio file
                    import torchaudio
                    target_length = int(32000 * 3.0)  # 3 seconds at 32kHz
                    try:
                        audio, sr = torchaudio.load(str(no_bird_file))
                        # Convert to mono if needed
                        if audio.shape[0] > 1:
                            audio = audio.mean(dim=0, keepdim=False)
                        else:
                            audio = audio.squeeze(0)
                        
                        # Ensure consistent shape and length
                        audio = audio.squeeze()  # Remove any extra dimensions
                        if len(audio) > target_length:
                            audio = audio[:target_length]  # Truncate
                        elif len(audio) < target_length:
                            # Pad with zeros
                            padding = target_length - len(audio)
                            audio = torch.nn.functional.pad(audio, (0, padding))
                        
                        # Ensure 1D tensor
                        audio = audio.view(-1)  # Force to 1D shape [96000]
                            
                    except Exception as e:
                        logger.warning(f"Error loading {no_bird_file}: {e}")
                        # Create silence as fallback
                        audio = torch.zeros(target_length, dtype=torch.float32)
                    
                    if self.bird_dataset.return_metadata:
                        metadata = {
                            'original_filename': no_bird_file.name,
                            'class_name': 'no_bird',
                            'file_path': str(no_bird_file)
                        }
                        return audio, self.no_bird_class_idx, metadata
                    else:
                        return audio, self.no_bird_class_idx
            
            def get_classes(self):
                """Return list of all class names including no_bird."""
                # Get bird classes in order
                bird_classes = [self.idx_to_class[i] for i in range(len(self.bird_dataset.class_to_idx))]
                bird_classes.append('no_bird')  # Add no_bird as last class
                return bird_classes
            
            def get_soft_labels_info(self):
                """Return soft labels info for the combined dataset."""
                return {
                    'total_files_with_soft_labels': 0,  # Will be set by distillation wrapper
                    'num_classes': 71,  # 70 bird + 1 no_bird
                    'metadata': {'num_classes': 71}
                }
            
            @property 
            def samples(self):
                """Provide samples list for compatibility."""
                # Create samples list for bird dataset
                bird_samples = []
                if hasattr(self.bird_dataset, 'samples'):
                    bird_samples = self.bird_dataset.samples
                
                # Add no_bird samples
                no_bird_samples = []
                for file_path in self.no_bird_files:
                    no_bird_samples.append({
                        'file_path': file_path,
                        'class_name': 'no_bird',
                        'class_idx': self.no_bird_class_idx,
                        'original_filename': file_path.name
                    })
                
                return bird_samples + no_bird_samples
        
        # Create combined dataset
        combined_dataset = CombinedPreprocessedDataset(bird_dataset, no_bird_files)
        
        # Create distillation wrapper that adds soft labels
        distillation_dataset = PreprocessedDistillationDataset(
            preprocessed_dataset=combined_dataset,
            soft_labels_path=self.soft_labels_path
        )
        
        # Create dataloader
        batch_size = dataset_config.get('batch_size', 32)
        num_workers = dataset_config.get('num_workers', 4)
        shuffle = (split == 'train')
        
        dataloader = DataLoader(
            distillation_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == 'train')  # Drop last batch only for training
        )
        
        logger.info(f"Created combined preprocessed distillation dataloader for {split}:")
        logger.info(f"  Bird samples: {len(bird_dataset)}")
        logger.info(f"  No_bird samples: {len(no_bird_files)}")
        logger.info(f"  Total samples: {len(distillation_dataset)}")
        logger.info(f"  Total classes: {len(combined_dataset.class_to_idx)}")
        
        return dataloader, distillation_dataset
    
    def setup_model(self):
        """Setup student model"""
        logger.info("Setting up student model...")
        
        # Create model with config parameters
        model_config = self.config.model
        
        # Convert matchbox config to regular dict to avoid Hydra struct issues
        matchbox_config = OmegaConf.to_container(model_config.get('matchbox', {}), resolve=True)
        
        self.model = Improved_Phi_GRU_ATT(
            num_classes=self.num_classes,
            spectrogram_type=model_config.get('spectrogram_type', 'mel'),
            sample_rate=self.config.dataset.get('sample_rate', 32000),
            n_mel_bins=model_config.get('n_mel_bins', 64),
            n_linear_filters=model_config.get('n_linear_filters', 64),
            f_min=self.config.dataset.get('lowcut', 0.0),
            f_max=self.config.dataset.get('highcut', 16000.0),
            hidden_dim=model_config.get('hidden_dim', 32),
            n_fft=model_config.get('n_fft', 1024),
            hop_length=model_config.get('hop_length', 320),
            matchbox=matchbox_config,
            breakpoint=model_config.get('initial_breakpoint', 4000.0),
            transition_width=model_config.get('initial_transition_width', 100.0)
        )
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Student model parameters: {total_params:,}")
        
        return self.model
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler, with optional different learning rates for filter parameters"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Check if we need different learning rates for filter parameters
        filter_lr_mult = self.config.get('filter_lr_multiplier', 1.0)
        
        if filter_lr_mult != 1.0:
            # Separate filter parameters from other parameters
            filter_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if any(filter_name in name for filter_name in ['breakpoint', 'transition_width', 'filter_bank']):
                    filter_params.append(param)
                    logger.info(f"Filter parameter: {name}, shape: {param.shape}")
                else:
                    other_params.append(param)
            
            if filter_params:
                # Get base learning rate from config
                base_lr = self.config.optimizer.get('lr', 0.001)
                filter_lr = base_lr * filter_lr_mult
                
                # Create optimizer with parameter groups
                param_groups = [
                    {'params': other_params, 'lr': base_lr},
                    {'params': filter_params, 'lr': filter_lr}
                ]
                
                # Create optimizer with parameter groups instead of using hydra
                optimizer_class = getattr(torch.optim, self.config.optimizer._target_.split('.')[-1])
                optimizer_kwargs = {k: v for k, v in self.config.optimizer.items() if k != '_target_'}
                optimizer_kwargs.pop('lr', None)  # Remove base lr since we're using param groups
                
                self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)
                
                logger.info(f"Using different learning rates: Base={base_lr}, Filter={filter_lr}")
            else:
                # No filter parameters found, use standard setup
                self.optimizer = hydra.utils.instantiate(
                    self.config.optimizer,
                    params=self.model.parameters()
                )
                logger.info("No filter parameters found, using standard optimizer setup")
        else:
            # Standard optimizer setup
            self.optimizer = hydra.utils.instantiate(
                self.config.optimizer,
                params=self.model.parameters()
            )
            logger.info(f"Using single learning rate: {self.config.optimizer.get('lr', 'default')}")
        
        # Scheduler (if specified)
        if 'scheduler' in self.config:
            self.scheduler = hydra.utils.instantiate(
                self.config.scheduler,
                optimizer=self.optimizer
            )
        
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        if self.scheduler:
            logger.info(f"Scheduler: {type(self.scheduler).__name__}")
    
    def setup_criterion(self):
        """Setup loss function based on configuration"""
        logger.info("Setting up loss function...")
        
        # Get loss configuration
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'distillation')  # Default to distillation
        
        # Common parameters
        alpha = self.config.distillation.get('alpha', 0.5)
        temperature = self.config.distillation.get('temperature', 4.0)
        adaptive = self.config.distillation.get('adaptive', False)
        
        # Loss-specific parameters
        gamma = loss_config.get('gamma', 2.0)  # For focal loss
        class_weights = loss_config.get('class_weights', None)  # Per-class weights
        
        logger.info(f"Loss type: {loss_type}")
        logger.info(f"Alpha (soft weight): {alpha}")
        logger.info(f"Temperature: {temperature}")
        logger.info(f"Adaptive: {adaptive}")
        
        if loss_type == 'distillation':
            # Standard knowledge distillation
            if adaptive:
                self.criterion = AdaptiveDistillationLoss(
                    alpha=alpha,
                    temperature=temperature,
                    adaptation_rate=self.config.distillation.get('adaptation_rate', 0.1)
                )
            else:
                self.criterion = DistillationLoss(
                    alpha=alpha,
                    temperature=temperature
                )
            logger.info("Using standard DistillationLoss")
            
        elif loss_type == 'focal':
            # Pure focal loss (no distillation)
            logger.info(f"Gamma (focal): {gamma}")
            
            if class_weights == 'auto':
                # Automatically compute class weights from training data
                logger.info("Computing automatic class weights...")
                # We'll compute these after data loading
                self.criterion = None  # Will be set after data statistics
                self._setup_focal_with_auto_weights = True
            else:
                self.criterion = FocalLoss(
                    alpha=class_weights if class_weights is not None else 1.0,
                    gamma=gamma
                )
                self._setup_focal_with_auto_weights = False
            logger.info("Using FocalLoss")
            
        elif loss_type == 'focal_distillation':
            # Focal loss + knowledge distillation
            logger.info(f"Gamma (focal): {gamma}")
            
            if class_weights == 'auto':
                # Automatically compute class weights from training data
                logger.info("Computing automatic class weights for focal_distillation...")
                # We'll compute these after data loading
                self.criterion = None  # Will be set after data statistics
                self._setup_focal_distillation_with_auto_weights = True
                self._focal_distillation_params = {
                    'alpha': alpha,
                    'gamma': gamma, 
                    'temperature': temperature,
                    'adaptive': adaptive,
                    'adaptation_rate': loss_config.get('adaptation_rate', 0.1)
                }
            else:
                if adaptive:
                    self.criterion = AdaptiveFocalDistillationLoss(
                        alpha=alpha,
                        gamma=gamma,
                        temperature=temperature,
                        class_weights=class_weights,
                        adaptation_rate=loss_config.get('adaptation_rate', 0.1)
                    )
                    logger.info("Using AdaptiveFocalDistillationLoss")
                else:
                    self.criterion = FocalDistillationLoss(
                        alpha=alpha,
                        gamma=gamma,
                        temperature=temperature,
                        class_weights=class_weights
                    )
                    logger.info("Using FocalDistillationLoss")
                self._setup_focal_distillation_with_auto_weights = False
                
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. "
                           f"Supported: 'distillation', 'focal', 'focal_distillation'")
        
        # Store loss type for later reference
        self.loss_type = loss_type
    
    def _load_cached_class_weights(self):
        """Load class weights from cache if available and recent"""
        weights_cache_path = self.output_dir / "computed_class_weights.json"
        
        if not weights_cache_path.exists():
            return None
            
        try:
            import json
            with open(weights_cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is recent (less than 24 hours old)
            cache_age_hours = (time.time() - cache_data.get('timestamp', 0)) / 3600
            max_cache_age = self.config.get('loss', {}).get('cache_max_age_hours', 24)
            
            if cache_age_hours > max_cache_age:
                logger.info(f"Cache is {cache_age_hours:.1f} hours old (max: {max_cache_age}), will recompute weights")
                return None
            
            class_weights = cache_data.get('class_weights')
            if class_weights and len(class_weights) == self.num_classes:
                logger.info(f"Loaded cached class weights: {[f'{w:.3f}' for w in class_weights]}")
                logger.info(f"Cache info: {cache_data.get('samples_used', 'unknown')} samples from {cache_data.get('total_dataset_size', 'unknown')} total")
                return class_weights
            else:
                logger.warning("Cached weights don't match current number of classes, will recompute")
                return None
                
        except Exception as e:
            logger.warning(f"Could not load weights cache: {e}")
            return None
    
    def _compute_automatic_class_weights(self, train_dataset):
        """Compute class weights automatically from training dataset with fast sampling"""
        logger.info("Computing automatic class weights from training data...")
        
        # Try to load from cache first
        cached_weights = self._load_cached_class_weights()
        if cached_weights is not None:
            class_weights = cached_weights
        else:
            # Get loss configuration for sampling parameters
            loss_config = self.config.get('loss', {})
            
            # Fast sampling parameters
            max_samples = loss_config.get('weight_calculation_samples', 10000)  # !!! DA SISTEMARE: E' HARDCODED
            use_sampling = loss_config.get('use_fast_sampling', True)  # Enable fast sampling by default
            
            dataset_size = len(train_dataset)
            
            if not use_sampling or dataset_size <= max_samples:
                # Use full dataset if sampling disabled or dataset is small
                samples_to_scan = dataset_size
                indices = range(dataset_size)
                logger.info(f"Scanning full dataset ({dataset_size} samples) to compute class distribution...")
            else:
                # Use statistical sampling for faster computation
                samples_to_scan = min(max_samples, dataset_size)
                indices = np.random.choice(dataset_size, size=samples_to_scan, replace=False)
                logger.info(f"Using fast sampling: scanning {samples_to_scan}/{dataset_size} samples ({samples_to_scan/dataset_size*100:.1f}%)")
            
            # Get class distribution from sampled dataset
            class_counts = {}
            
            pbar = tqdm(indices, desc="Computing class weights", unit="samples")
            successful_samples = 0
            
            for i in pbar:
                try:
                    _, hard_label, soft_label = train_dataset[i]
                    if isinstance(hard_label, torch.Tensor):
                        label = hard_label.item()
                    else:
                        label = hard_label
                    
                    class_counts[label] = class_counts.get(label, 0) + 1
                    successful_samples += 1
                    
                    # Update progress bar every 50 samples (faster updates)
                    if successful_samples % 50 == 0:
                        pbar.set_postfix(classes_found=len(class_counts), scanned=successful_samples)
                        
                except Exception as e:
                    # Handle corrupted files gracefully and continue
                    logger.debug(f"Failed to load sample {i}: {e}")
                    continue
            
            # IMPORTANT: Also scan soft labels to find missing classes (like non-bird)
            logger.info("Scanning soft labels to identify additional classes not in preprocessed dataset...")
            try:
                soft_labels_info = train_dataset.get_soft_labels_info()
                total_expected_classes = soft_labels_info.get('num_classes', self.num_classes)
                logger.info(f"Soft labels info retrieved: expected classes = {total_expected_classes}")
                logger.info(f"Expected total classes: {total_expected_classes}, Found in dataset: {len(class_counts)}")
            except Exception as e:
                logger.error(f"Error getting soft labels info: {e}")
                total_expected_classes = self.num_classes
            
            # Check if we found all expected classes
            missing_classes = []
            for expected_class_idx in range(total_expected_classes):
                if expected_class_idx not in class_counts:
                    missing_classes.append(expected_class_idx)
            
            logger.info(f"Missing classes analysis: Expected={total_expected_classes}, Found={list(class_counts.keys())}, Missing={missing_classes}")
            
            if missing_classes:
                logger.info(f"Found {len(missing_classes)} missing classes in preprocessed dataset: {missing_classes}")
                
                # FIXED: Use simple, reliable estimates for missing classes
                for cls_idx in missing_classes:
                    if cls_idx == 70:  # Non-bird class
                        # Based on user info: ~900 non-bird samples in the dataset
                        estimated_count = 900
                        logger.info(f"Using fixed estimate for non-bird class {cls_idx}: {estimated_count} samples")
                    elif cls_idx == 44:  # Pernis apivorus (very rare bird)
                        # Rare bird species - conservative estimate
                        estimated_count = max(1, dataset_size // (total_expected_classes * 100))  # Very small
                        logger.info(f"Using conservative estimate for rare bird class {cls_idx}: {estimated_count} samples")
                    else:
                        # Other missing classes - moderate estimate
                        estimated_count = max(1, dataset_size // (total_expected_classes * 50))
                        logger.info(f"Using moderate estimate for missing class {cls_idx}: {estimated_count} samples")
                    
                    class_counts[cls_idx] = estimated_count
                
                logger.info(f"Successfully estimated counts for missing classes with fixed values: {missing_classes}")
                logger.info(f"Updated class_counts: {dict(sorted(class_counts.items()))}")
            else:
                logger.info("No missing classes detected - all expected classes found in dataset")
            
            if successful_samples == 0:
                logger.error("No samples could be loaded for class weight computation!")
                # Return equal weights as fallback
                num_classes = total_expected_classes
                class_weights = [1.0] * num_classes
            else:
                # Convert to ordered list - USE THE TOTAL EXPECTED CLASSES
                expected_num_classes = total_expected_classes  # Use the actual number from soft labels
                max_class = max(max(class_counts.keys()) if class_counts else 0, expected_num_classes - 1)
                class_counts_list = [class_counts.get(i, 0) for i in range(max_class + 1)]
                
                # Ensure we have exactly the expected number of classes
                while len(class_counts_list) < expected_num_classes:
                    class_counts_list.append(0)  # Add missing classes with 0 count
                
                # FIXED: Handle scaling carefully - don't scale estimated classes twice
                if use_sampling and samples_to_scan < dataset_size:
                    scaling_factor = dataset_size / samples_to_scan
                    scaled_class_counts = []
                    
                    for i, count in enumerate(class_counts_list):
                        if i in missing_classes and count > 0:
                            # Don't scale estimated classes - they're already at reasonable values
                            scaled_class_counts.append(count)
                            logger.debug(f"Class {i} using fixed estimate (no scaling): {count}")
                        else:
                            # Scale sampled classes
                            scaled_count = int(count * scaling_factor)
                            scaled_class_counts.append(scaled_count)
                            
                    logger.info(f"Applied selective scaling by {scaling_factor:.2f} (skipped fixed estimates)")
                    logger.info(f"Estimated class distribution: {scaled_class_counts}")
                else:
                    scaled_class_counts = class_counts_list
                    logger.info(f"Actual class distribution: {class_counts_list}")
                
                # Compute class weights using inverse frequency - FIXED CALCULATION
                # Use the actual dataset size, not the scaled total
                total_samples_actual = dataset_size  # Use real dataset size
                num_classes = len(scaled_class_counts)
                
                # Handle missing classes with smart fallback
                min_count_threshold = max(1, total_samples_actual // (num_classes * 100))  # More conservative minimum
                class_weights = []
                for i, count in enumerate(scaled_class_counts):
                    if count == 0:
                        # For missing classes, assign moderate weight based on average rare class frequency
                        # This avoids both zero division and extreme weights
                        non_zero_counts = [c for c in scaled_class_counts if c > 0]
                        if non_zero_counts:
                            # Use smallest non-zero count as estimate for missing class
                            effective_count = min(non_zero_counts)
                        else:
                            effective_count = total_samples_actual // (num_classes * 10)  # Very conservative fallback
                        logger.warning(f"Class {i} not found in sample - estimating with count {effective_count}")
                    else:
                        effective_count = max(count, min_count_threshold)  # Avoid zero division
                    
                    # Standard inverse frequency weighting formula
                    weight = total_samples_actual / (num_classes * effective_count)
                    class_weights.append(weight)
            
            # Apply scaling
            alpha_scaling = loss_config.get('alpha_scaling', 1.0)
            class_weights = [w * alpha_scaling for w in class_weights]
            
            logger.info(f"Computed class weights: {[f'{w:.3f}' for w in class_weights]}")
            
            # Save computed weights to cache for future use
            weights_cache_path = self.output_dir / "computed_class_weights.json"
            try:
                import json
                cache_data = {
                    'class_weights': class_weights,
                    'class_distribution': scaled_class_counts if 'scaled_class_counts' in locals() else class_counts_list,
                    'missing_classes_estimated': missing_classes if 'missing_classes' in locals() else [],
                    'samples_used': successful_samples,
                    'total_dataset_size': dataset_size,
                    'use_sampling': use_sampling,
                    'timestamp': time.time()
                }
                with open(weights_cache_path, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"Saved computed weights to cache: {weights_cache_path}")
            except Exception as e:
                logger.warning(f"Could not save weights cache: {e}")
        
        # Create appropriate loss function
        if hasattr(self, '_setup_focal_with_auto_weights') and self._setup_focal_with_auto_weights:
            # Pure focal loss
            gamma = loss_config.get('gamma', 2.0)
            self.criterion = FocalLoss(
                alpha=class_weights,
                gamma=gamma
            )
            logger.info("Automatic class weights applied to FocalLoss")
            
        elif hasattr(self, '_setup_focal_distillation_with_auto_weights') and self._setup_focal_distillation_with_auto_weights:
            # Focal distillation loss
            params = self._focal_distillation_params
            
            if params['adaptive']:
                self.criterion = AdaptiveFocalDistillationLoss(
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    temperature=params['temperature'],
                    class_weights=class_weights,
                    adaptation_rate=params['adaptation_rate']
                )
                logger.info("Automatic class weights applied to AdaptiveFocalDistillationLoss")
            else:
                self.criterion = FocalDistillationLoss(
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    temperature=params['temperature'],
                    class_weights=class_weights
                )
                logger.info("Automatic class weights applied to FocalDistillationLoss")
    
    def save_best_model(self):
        """Saves the best model based on validation loss."""
        model_path = self.output_dir / f"{self.config.experiment_name}_best_model.pth"
        logger.info(f"Saving best model to {model_path}...")
        torch.save(self.model.state_dict(), model_path)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]", unit="batch", disable=False)
        batch_count = 0
        total_batches = len(self.train_loader)
        
        for audio, hard_labels, soft_labels in pbar:
            batch_count += 1
            # Move to device
            audio = audio.to(self.device)
            hard_labels = hard_labels.to(self.device)
            soft_labels = soft_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio)
            
            # Compute loss based on loss type
            if hasattr(self, 'loss_type') and self.loss_type == 'focal':
                # Pure focal loss (no distillation)
                loss = self.criterion(logits, hard_labels)
                hard_loss = loss  # For focal loss, the returned value is the hard loss
                soft_loss = torch.tensor(0.0, device=self.device)  # No soft loss
            else:
                # Distillation-based losses (returns tuple)
                loss, hard_loss, soft_loss = self.criterion(logits, hard_labels, soft_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            # Accuracy (based on hard labels)
            _, predicted = logits.max(1)
            total += hard_labels.size(0)
            correct += predicted.eq(hard_labels).sum().item()
            
            # Set postfix for tqdm progress bar
            current_acc = (100. * correct / total)
            if hasattr(self, 'loss_type') and self.loss_type == 'focal':
                pbar.set_postfix(
                    batch=f"{batch_count}/{total_batches}",
                    loss=f"{loss.item():.4f}", 
                    focal_loss=f"{hard_loss.item():.4f}",
                    acc=f"{current_acc:.2f}%"
                )
            else:
                pbar.set_postfix(
                    batch=f"{batch_count}/{total_batches}",
                    loss=f"{loss.item():.4f}", 
                    hard_loss=f"{hard_loss.item():.4f}", 
                    soft_loss=f"{soft_loss.item():.4f}",
                    acc=f"{current_acc:.2f}%"
                )
            
            # Log progress every 10 batches
            if batch_count % 10 == 0:
                logger.info(f"Epoch {self.epoch} - Batch {batch_count}/{total_batches} - Loss: {loss.item():.4f} - Acc: {current_acc:.2f}%")
        
        avg_loss = total_loss / len(self.train_loader)
        avg_hard_loss = total_hard_loss / len(self.train_loader)
        avg_soft_loss = total_soft_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Store metrics
        self.train_losses.append(avg_loss)
        self.hard_losses.append(avg_hard_loss)
        self.soft_losses.append(avg_soft_loss)
        self.train_accs.append(accuracy)
        
        logger.info(f"Epoch {self.epoch} [Train] Avg Loss: {avg_loss:.4f}, Avg Acc: {accuracy:.4f}")
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Add explicit garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]", unit="batch", disable=False)
        batch_count = 0
        total_batches = len(self.val_loader)
        
        logger.info(f"Starting validation with {total_batches} batches")
        
        with torch.no_grad():
            try:
                for batch_idx, (audio, hard_labels, soft_labels) in enumerate(pbar):
                    batch_count += 1
                    logger.debug(f"Processing validation batch {batch_count}/{total_batches}")
                    
                    # Move to device with error handling
                    try:
                        audio = audio.to(self.device)
                        hard_labels = hard_labels.to(self.device)
                        soft_labels = soft_labels.to(self.device)
                    except Exception as e:
                        logger.error(f"Error moving batch {batch_count} to device: {e}")
                        continue
                    
                    # Forward pass with error handling
                    try:
                        logits = self.model(audio)
                    except Exception as e:
                        logger.error(f"Error in forward pass for batch {batch_count}: {e}")
                        continue
                    
                    # Compute loss based on loss type
                    try:
                        if hasattr(self, 'loss_type') and self.loss_type == 'focal':
                            # Pure focal loss (no distillation)
                            loss = self.criterion(logits, hard_labels)
                        else:
                            # Distillation-based losses (returns tuple)
                            loss, _, _ = self.criterion(logits, hard_labels, soft_labels)
                    except Exception as e:
                        logger.error(f"Error computing loss for batch {batch_count}: {e}")
                        continue
                    
                    # Statistics
                    total_loss += loss.item()
                    
                    # Accuracy
                    _, predicted = logits.max(1)
                    total += hard_labels.size(0)
                    correct += predicted.eq(hard_labels).sum().item()
            
                    current_acc = (100. * correct / total)
                    pbar.set_postfix(
                        batch=f"{batch_count}/{total_batches}",
                        loss=f"{loss.item():.4f}", 
                        acc=f"{current_acc:.2f}%"
                    )
                    
                    # Log progress every 5 batches to identify where it stops
                    if batch_count % 5 == 0:
                        logger.info(f"Validation batch {batch_count}/{total_batches} completed - Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%")
                    
                    # Force garbage collection every 10 batches to prevent memory issues
                    if batch_count % 10 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
            except Exception as e:
                logger.error(f"Critical error during validation at batch {batch_count}: {e}")
                logger.error(f"Exception type: {type(e).__name__}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Return partial results if we have any
                if total > 0:
                    avg_loss = total_loss / max(1, batch_count)
                    accuracy = 100. * correct / total
                    logger.warning(f"Returning partial validation results: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
                    return avg_loss, accuracy
                else:
                    logger.error("No validation data processed, returning default values")
                    return float('inf'), 0.0
        
        logger.info(f"Validation completed successfully with {batch_count} batches")
        
        # Calculate average loss and accuracy (convert to percentage like train_acc)
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            accuracy = 100. * correct / total if total > 0 else 0.0
        else:
            avg_loss = float('inf')
            accuracy = 0.0
        
        # Store metrics
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        logger.info(f"Epoch {self.epoch} [Val] Avg Loss: {avg_loss:.4f}, Avg Acc: {accuracy:.4f}")
        return avg_loss, accuracy
    
    def train(self):
        """Main training loop"""
        # Setup components before training
        train_dataset, _, _ = self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        
        # If criterion is None, we need to compute automatic class weights
        if self.criterion is None:
            self._compute_automatic_class_weights(train_dataset)
        
        # Early stopping
        early_stopper = EarlyStopping(
            patience=self.config.training.get('patience', 10),
            min_delta=self.config.training.get('min_delta', 0.001)
        )
        
        logger.info("Starting training loop...")
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch + 1
            
            # Track learning rate at the start of the epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Optional pause between training and validation to reduce CPU stress
            validation_pause = self.config.training.get('validation_pause', 0)
            if validation_pause > 0:
                logger.info(f"Pausing {validation_pause} seconds between training and validation to reduce CPU load...")
                time.sleep(validation_pause)
            
            # Train and validate
            self.train_epoch()
            val_loss, val_acc = self.validate_epoch()
            
            # Track filter parameters
            self._log_filter_parameters()
            
            # Monitor fully learnable filters if enabled
            if (self.config.get('logging', {}).get('monitor_filters', False) and 
                self.config.model.get('spectrogram_type') == 'fully_learnable'):
                self._log_fully_learnable_filter_stats()
                
                # Save filter visualization periodically
                viz_interval = self.config.get('logging', {}).get('filter_visualization_interval', 10)
                if viz_interval > 0 and self.epoch % viz_interval == 0:
                    self._save_filter_visualization()
            
            # Update learning rate scheduler
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f}")
                self.save_best_model()
            
            # Early stopping check
            if early_stopper(val_loss):
                logger.info("Early stopping triggered.")
                break
        
        logger.info("Training loop finished.")
        self.save_training_plots()
    
    def _log_filter_parameters(self):
        """Log filter parameters for both semi-learnable and fully learnable filters."""
        if (self.config.model.get('spectrogram_type') == 'combined_log_linear' and 
            hasattr(self.model, 'combined_log_linear_spec') and 
            self.model.combined_log_linear_spec is not None):
            # Semi-learnable filter parameters
            current_breakpoint = self.model.combined_log_linear_spec.breakpoint.item()
            current_transition_width = self.model.combined_log_linear_spec.transition_width.item()
            self.breakpoint_history.append(current_breakpoint)
            self.transition_width_history.append(current_transition_width)
            logger.info(f"Epoch {self.epoch} - Filter params: Breakpoint={current_breakpoint:.2f}Hz, Transition Width={current_transition_width:.2f}")
    
    def _log_fully_learnable_filter_stats(self):
        """Log statistics for fully learnable filter bank."""
        if not (hasattr(self.model, 'combined_log_linear_spec') and 
                hasattr(self.model.combined_log_linear_spec, 'filter_bank')):
            return
            
        filter_matrix = self.model.combined_log_linear_spec.filter_bank  # [64, 513]
        
        # Basic statistics
        stats = {
            'mean': filter_matrix.mean().item(),
            'std': filter_matrix.std().item(),
            'min': filter_matrix.min().item(),
            'max': filter_matrix.max().item(),
            'total_params': filter_matrix.numel(),
            'per_filter_variance': filter_matrix.var(dim=1).mean().item(),
        }
        
        # Cross-filter similarity (detect degeneracy)
        similarity = self._compute_filter_similarity(filter_matrix)
        stats['cross_filter_similarity'] = similarity
        
        # Gradient analysis if available
        if filter_matrix.grad is not None:
            stats['grad_norm'] = filter_matrix.grad.norm().item()
        else:
            stats['grad_norm'] = 0.0
        
        # Log the statistics
        log_msg = f"Epoch {self.epoch} - Filter Bank Stats: "
        log_msg += f"Mean={stats['mean']:.4f}, Std={stats['std']:.4f}, "
        log_msg += f"Min={stats['min']:.4f}, Max={stats['max']:.4f}, "
        log_msg += f"Similarity={stats['cross_filter_similarity']:.4f}, "
        log_msg += f"GradNorm={stats['grad_norm']:.4f}"
        
        logger.info(log_msg)
        
        # Store in history for plotting (if needed)
        if not hasattr(self, 'filter_stats_history'):
            self.filter_stats_history = []
        self.filter_stats_history.append(stats)
    
    def _compute_filter_similarity(self, filter_matrix):
        """Compute average cosine similarity between filters (detect degeneracy)."""
        import torch.nn.functional as F
        
        # Normalize filters
        normalized = F.normalize(filter_matrix, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(normalized, normalized.t())
        
        # Get off-diagonal elements (exclude self-similarity)
        mask = ~torch.eye(filter_matrix.size(0), dtype=torch.bool, device=filter_matrix.device)
        off_diagonal_similarities = similarity_matrix[mask]
        
        return off_diagonal_similarities.mean().item()
    
    def _save_filter_visualization(self):
        """Save comprehensive visualization of learned filters with evolution tracking."""
        if not (hasattr(self.model, 'combined_log_linear_spec') and 
                hasattr(self.model.combined_log_linear_spec, 'filter_bank')):
            return
        
        try:
            import matplotlib.pyplot as plt
            
            filter_matrix = self.model.combined_log_linear_spec.filter_bank.detach().cpu().numpy()
            
            # Create visualization directory
            viz_dir = self.output_dir / "filter_visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Store filter snapshots for evolution tracking
            if not hasattr(self, 'filter_snapshots'):
                self.filter_snapshots = []
            self.filter_snapshots.append(filter_matrix.copy())
            
            # 1. COMPOSITE: Single 8-subplot visualization of current state
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(f'Fully Learnable Filter Bank - Epoch {self.epoch}', fontsize=16)
            axes = axes.flatten()
            
            # Plot first 8 filters with enhanced details
            for i in range(min(8, filter_matrix.shape[0])):
                axes[i].plot(filter_matrix[i], linewidth=2, alpha=0.8)
                axes[i].set_title(f'Filter {i+1}\n(Mean: {filter_matrix[i].mean():.3f}, Std: {filter_matrix[i].std():.3f})')
                axes[i].set_xlabel('Frequency Bin')
                axes[i].set_ylabel('Weight')
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim([filter_matrix.min() * 1.1, filter_matrix.max() * 1.1])
            
            # Hide unused subplots
            for i in range(8, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(viz_dir / "current_filters_composite.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            # 2. EVOLUTION: Multi-epoch filter evolution (only if we have multiple snapshots)
            if len(self.filter_snapshots) > 1:
                fig, axes = plt.subplots(2, 4, figsize=(20, 12))
                fig.suptitle(f'Filter Evolution Over Training (Epochs 1-{self.epoch})', fontsize=16)
                axes = axes.flatten()
                
                # Create colormap for epochs
                import matplotlib.cm as cm
                colors = cm.viridis(np.linspace(0, 1, len(self.filter_snapshots)))
                
                for i in range(min(8, filter_matrix.shape[0])):
                    for epoch_idx, snapshot in enumerate(self.filter_snapshots):
                        alpha = 0.3 + 0.7 * (epoch_idx / (len(self.filter_snapshots) - 1))  # Fade older epochs
                        label = f'Epoch {epoch_idx + 1}' if epoch_idx in [0, len(self.filter_snapshots)-1] else None
                        axes[i].plot(snapshot[i], color=colors[epoch_idx], alpha=alpha, 
                                   linewidth=2 if epoch_idx == len(self.filter_snapshots)-1 else 1,
                                   label=label)
                    
                    axes[i].set_title(f'Filter {i+1} Evolution')
                    axes[i].set_xlabel('Frequency Bin')
                    axes[i].set_ylabel('Weight')
                    axes[i].grid(True, alpha=0.3)
                    if i == 0:  # Add legend only to first subplot
                        axes[i].legend()
                
                # Hide unused subplots
                for i in range(8, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                plt.savefig(viz_dir / "filter_evolution_composite.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # 3. HEATMAP: Complete filter bank visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Current state heatmap
            im1 = ax1.imshow(filter_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
            ax1.set_title(f'All 64 Filters - Epoch {self.epoch}')
            ax1.set_xlabel('Frequency Bin')
            ax1.set_ylabel('Filter Index')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Statistics evolution (if available)
            if hasattr(self, 'filter_stats_history') and len(self.filter_stats_history) > 1:
                epochs_so_far = list(range(1, len(self.filter_stats_history) + 1))
                means = [stats['mean'] for stats in self.filter_stats_history]
                stds = [stats['std'] for stats in self.filter_stats_history]
                similarities = [stats['cross_filter_similarity'] for stats in self.filter_stats_history]
                grad_norms = [stats['grad_norm'] for stats in self.filter_stats_history]
                
                ax2_twin1 = ax2.twinx()
                ax2_twin2 = ax2.twinx()
                ax2_twin2.spines['right'].set_position(('outward', 60))
                
                line1 = ax2.plot(epochs_so_far, means, 'b.-', linewidth=2, label='Mean', markersize=6)
                line2 = ax2_twin1.plot(epochs_so_far, similarities, 'r.-', linewidth=2, label='Similarity', markersize=6)
                line3 = ax2_twin2.plot(epochs_so_far, grad_norms, 'g.-', linewidth=2, label='Grad Norm', markersize=6)
                
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Filter Mean', color='b')
                ax2_twin1.set_ylabel('Cross-Filter Similarity', color='r')
                ax2_twin2.set_ylabel('Gradient Norm', color='g')
                
                ax2.tick_params(axis='y', labelcolor='b')
                ax2_twin1.tick_params(axis='y', labelcolor='r')
                ax2_twin2.tick_params(axis='y', labelcolor='g')
                
                ax2.set_title('Filter Statistics Evolution')
                ax2.grid(True, alpha=0.3)
                
                # Combine legends
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax2.legend(lines, labels, loc='upper right')
            else:
                ax2.text(0.5, 0.5, f'Statistics tracking started.\nComplete evolution available\nafter multiple epochs.', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12,
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5))
                ax2.set_title('Filter Statistics Evolution (Pending)')
            
            plt.tight_layout()
            plt.savefig(viz_dir / "filter_analysis_composite.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Enhanced filter visualizations saved for epoch {self.epoch}")
            
        except ImportError:
            logger.warning("matplotlib not available, skipping filter visualization")
        except Exception as e:
            logger.error(f"Error saving filter visualization: {e}")
    
    def _generate_fully_learnable_summary(self):
        """Generate comprehensive summary of fully learnable filter evolution."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            logger.info("Generating fully learnable filter summary...")
            
            # Create summary directory
            summary_dir = self.output_dir / "fully_learnable_summary"
            summary_dir.mkdir(exist_ok=True)
            
            # Final filter state analysis
            final_filters = self.filter_snapshots[-1]  # [64, 513]
            
            # 1. COMPREHENSIVE FILTER SHOWCASE (3x3 grid of 9 representative filters)
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle('Representative Fully Learnable Filters - Final State', fontsize=18)
            
            # Select 9 diverse filters (every 7th filter to get good coverage)
            selected_filters = [i * 7 for i in range(9) if i * 7 < final_filters.shape[0]]
            
            for idx, filter_idx in enumerate(selected_filters):
                row, col = idx // 3, idx % 3
                ax = axes[row, col]
                
                filter_weights = final_filters[filter_idx]
                ax.plot(filter_weights, linewidth=3, alpha=0.9, color='darkblue')
                ax.fill_between(range(len(filter_weights)), filter_weights, alpha=0.3, color='lightblue')
                
                # Add statistics
                mean_val = filter_weights.mean()
                std_val = filter_weights.std()
                peak_freq = np.argmax(filter_weights)
                
                ax.set_title(f'Filter {filter_idx+1}\nPeak @ bin {peak_freq} | μ={mean_val:.3f}, σ={std_val:.3f}', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Frequency Bin')
                ax.set_ylabel('Filter Weight')
                ax.grid(True, alpha=0.4)
                ax.set_ylim([final_filters.min() * 1.1, final_filters.max() * 1.1])
            
            plt.tight_layout()
            plt.savefig(summary_dir / "representative_filters_final.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            # 2. FILTER DIVERSITY ANALYSIS
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Fully Learnable Filter Bank Analysis', fontsize=16)
            
            # Diversity heatmap
            from scipy.spatial.distance import pdist, squareform
            import matplotlib.cm as cm
            
            # Compute pairwise correlations
            correlations = np.corrcoef(final_filters)
            
            im1 = ax1.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1)
            ax1.set_title('Filter Correlation Matrix\n(Diagonal = Self-correlation = 1)')
            ax1.set_xlabel('Filter Index')
            ax1.set_ylabel('Filter Index')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Filter peak frequencies
            peak_frequencies = np.argmax(final_filters, axis=1)
            ax2.hist(peak_frequencies, bins=20, alpha=0.7, edgecolor='black')
            ax2.set_title('Distribution of Filter Peak Frequencies')
            ax2.set_xlabel('Frequency Bin (Peak Response)')
            ax2.set_ylabel('Number of Filters')
            ax2.grid(True, alpha=0.3)
            
            # Filter selectivity (std as proxy for selectivity)
            filter_selectivity = np.std(final_filters, axis=1)
            ax3.scatter(range(len(filter_selectivity)), filter_selectivity, alpha=0.7, s=50)
            ax3.set_title('Filter Selectivity (Higher = More Selective)')
            ax3.set_xlabel('Filter Index')
            ax3.set_ylabel('Standard Deviation (Selectivity)')
            ax3.grid(True, alpha=0.3)
            
            # Evolution summary (if we have multiple snapshots)
            if len(self.filter_snapshots) > 1:
                epochs = list(range(1, len(self.filter_snapshots) + 1))
                
                # Track evolution of first 5 filters
                for i in range(min(5, final_filters.shape[0])):
                    evolution = [snapshot[i].std() for snapshot in self.filter_snapshots]
                    ax4.plot(epochs, evolution, '.-', linewidth=2, label=f'Filter {i+1}', alpha=0.8)
                
                ax4.set_title('Filter Selectivity Evolution')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Filter Selectivity (Std)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Evolution tracking\nrequires multiple epochs', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Filter Evolution (Insufficient Data)')
            
            plt.tight_layout()
            plt.savefig(summary_dir / "filter_analysis_comprehensive.png", dpi=200, bbox_inches='tight')
            plt.close()
            
            # 3. SAVE FILTER DATA FOR EXTERNAL ANALYSIS
            np.save(summary_dir / "final_filter_weights.npy", final_filters)
            
            # Create text summary
            summary_text = f"""
FULLY LEARNABLE FILTER BANK SUMMARY
===================================

Training Epochs: {len(self.filter_snapshots)}
Total Filter Parameters: {final_filters.size:,}
Filter Bank Shape: {final_filters.shape}

FINAL STATE STATISTICS:
- Mean Weight: {final_filters.mean():.6f}
- Std Weight: {final_filters.std():.6f}
- Min Weight: {final_filters.min():.6f}
- Max Weight: {final_filters.max():.6f}
- Weight Range: {final_filters.max() - final_filters.min():.6f}

FILTER DIVERSITY:
- Average Inter-Filter Correlation: {np.mean(correlations[np.triu_indices_from(correlations, k=1)]):.6f}
- Most Similar Filters: {np.unravel_index(np.argmax(correlations - np.eye(len(correlations))), correlations.shape)}
- Peak Frequency Spread: {np.std(peak_frequencies):.2f} bins

SPECIALIZATION ANALYSIS:
- Most Selective Filter: #{np.argmax(filter_selectivity)+1} (std={np.max(filter_selectivity):.6f})
- Least Selective Filter: #{np.argmin(filter_selectivity)+1} (std={np.min(filter_selectivity):.6f})
- Average Selectivity: {np.mean(filter_selectivity):.6f}

FILES GENERATED:
- representative_filters_final.png: 9 representative filters
- filter_analysis_comprehensive.png: Complete analysis
- final_filter_weights.npy: Raw filter weights for analysis

INTERPRETATION:
{'✅ Good filter diversity' if np.mean(correlations[np.triu_indices_from(correlations, k=1)]) < 0.1 else '⚠️ Some filter redundancy detected'}
{'✅ Healthy weight distribution' if 0.01 < final_filters.std() < 0.5 else '⚠️ Check weight initialization/learning rates'}
{'✅ Specialized filters developed' if np.std(filter_selectivity) > 0.01 else '⚠️ Filters may be too similar'}
"""
            
            with open(summary_dir / "filter_summary.txt", 'w') as f:
                f.write(summary_text)
            
            logger.info(f"Fully learnable filter summary saved to {summary_dir}")
            
        except Exception as e:
            logger.error(f"Error generating fully learnable summary: {e}")
    
    def test(self):
        """Test the model and generate metrics"""
        logger.info("Starting testing...")
        self.model.eval()
        
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for audio, hard_labels, soft_labels in tqdm(self.test_loader, desc="Testing"):
                # Move to device
                audio = audio.to(self.device)
                hard_labels = hard_labels.to(self.device)
                
                # Forward pass
                outputs = self.model(audio)
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions and labels
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(hard_labels.cpu().numpy())
        
        # Calculate metrics
        self.test_acc = sum(1 for i, j in zip(all_true, all_preds) if i == j) / len(all_true)
        logger.info(f"Test Accuracy: {self.test_acc:.4f}")
        
        # Get unique classes actually present in test set
        unique_classes_in_test = sorted(list(set(all_true)))
        actual_class_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" 
                             for i in unique_classes_in_test]
        
        logger.info(f"Classes present in test set: {unique_classes_in_test}")
        logger.info(f"Corresponding class names: {actual_class_names}")
        
        # Generate classification report with only actual classes
        report_str = classification_report(
            all_true, all_preds, 
            labels=unique_classes_in_test,
            target_names=actual_class_names, 
            zero_division=0
        )
        report_dict = classification_report(
            all_true, all_preds, 
            labels=unique_classes_in_test,
            target_names=actual_class_names, 
            zero_division=0, 
            output_dict=True
        )
        logger.info(f"Test Report:\n{report_str}")
        
        # Save confusion matrix using actual classes
        cm_png_path = self.output_dir / "confusion_matrix.png"
        cm_csv_path = self.output_dir / "confusion_matrix.csv"
        save_confusion_matrix(all_true, all_preds, actual_class_names, cm_png_path, cm_csv_path)
        
        # Save results and summary
        self.save_results(self.test_acc, report_dict)
        self.save_model_summary(report_str)
        return self.test_acc, report_dict
    
    def save_training_plots(self):
        """Saves training history plots."""
        if not self.train_losses:
            logger.warning("No training history to plot. Skipping plot generation.")
            return

        epochs = range(1, len(self.train_losses) + 1)
        
        # Determine if we have filter parameters to plot
        has_filter_params = (len(self.breakpoint_history) > 0 and 
                           len(self.transition_width_history) > 0 and 
                           len(self.breakpoint_history) == len(self.train_losses))
        
        # Adjust figure size and subplot layout based on available data
        if has_filter_params:
            plt.figure(figsize=(15, 12))
            subplot_rows, subplot_cols = 3, 2
        else:
            plt.figure(figsize=(12, 10))
            subplot_rows, subplot_cols = 2, 2
        
        # Plot Loss
        plt.subplot(subplot_rows, subplot_cols, 1)
        plt.plot(epochs, self.train_losses, '.-', label='Train Loss')
        plt.plot(epochs, self.val_losses, '.-', label='Val Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot Accuracy
        plt.subplot(subplot_rows, subplot_cols, 2)
        plt.plot(epochs, self.train_accs, '.-', label='Train Accuracy')
        plt.plot(epochs, self.val_accs, '.-', label='Val Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # Plot Hard/Soft Loss
        plt.subplot(subplot_rows, subplot_cols, 3)
        plt.plot(epochs, self.hard_losses, '.-', label='Hard Loss (Train)')
        plt.plot(epochs, self.soft_losses, '.-', label='Soft Loss (Train)')
        plt.title('Distillation Loss Components')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot Learning Rate
        plt.subplot(subplot_rows, subplot_cols, 4)
        plt.plot(epochs, self.learning_rates, '.-', label='Learning Rate')
        plt.title('Learning Rate')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        plt.legend()
        
        # Plot Filter Parameters (if available)
        if has_filter_params:
            # Plot Breakpoint Evolution
            plt.subplot(subplot_rows, subplot_cols, 5)
            plt.plot(epochs, self.breakpoint_history, '.-', label='Breakpoint (Hz)', color='green')
            plt.title('Differentiable Filter Breakpoint Evolution')
            plt.xlabel('Epochs')
            plt.ylabel('Breakpoint (Hz)')
            plt.grid(True)
            plt.legend()
            
            # Plot Transition Width Evolution
            plt.subplot(subplot_rows, subplot_cols, 6)
            plt.plot(epochs, self.transition_width_history, '.-', label='Transition Width', color='purple')
            plt.title('Differentiable Filter Transition Width Evolution')
            plt.xlabel('Epochs')
            plt.ylabel('Transition Width')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plot_path = self.output_dir / "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training plots to {plot_path}")
        
        # Save filter parameters evolution as separate CSV if available
        if has_filter_params:
            filter_params_path = self.output_dir / "filter_parameters_evolution.csv"
            import pandas as pd
            df = pd.DataFrame({
                'epoch': list(epochs),
                'breakpoint_hz': self.breakpoint_history,
                'transition_width': self.transition_width_history
            })
            df.to_csv(filter_params_path, index=False)
            logger.info(f"Saved filter parameters evolution to {filter_params_path}")
            
            # Generate advanced plots if available
            try:
                from pathlib import Path
                import sys
                sys.path.append(str(Path(__file__).parent))
                from advanced_plots import DistillationAnalyzer
                
                analyzer = DistillationAnalyzer(str(self.output_dir))
                advanced_plots = analyzer.generate_all_plots()
                logger.info(f"Generated {len(advanced_plots)} advanced analysis plots")
            except Exception as e:
                logger.warning(f"Could not generate advanced plots: {e}")
                
        # Generate fully learnable filter summary (if applicable)
        if (self.config.model.get('spectrogram_type') == 'fully_learnable' and 
            hasattr(self, 'filter_snapshots') and len(self.filter_snapshots) > 0):
            self._generate_fully_learnable_summary()

    def save_results(self, test_acc, report_dict):
        """Saves test results to a JSON file."""
        results = {
            "experiment_name": self.config.experiment_name,
            "best_val_acc": self.best_val_acc / 100.0,  # Convert percentage back to fraction for JSON
            "test_acc": test_acc,
            "test_f1_weighted": report_dict.get('weighted avg', {}).get('f1-score'),
            "test_precision_weighted": report_dict.get('weighted avg', {}).get('precision'),
            "test_recall_weighted": report_dict.get('weighted avg', {}).get('recall'),
            "total_params": sum(p.numel() for p in self.model.parameters()),
            "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        }
        
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Results saved to {results_path}")

    def save_model_summary(self, test_report_str):
        """Saves a summary of the model and training results to a text file."""
        summary_path = self.output_dir / "model_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Model: {self.model.__class__.__name__}\n")
            f.write(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}\n")
            f.write(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}\n\n")
            f.write(f"Best validation accuracy: {self.best_val_acc:.2f}%\n")
            f.write(f"Test accuracy: {self.test_acc*100:.2f}%\n")
            f.write(f"Classification Report (Test Set):\n{test_report_str}\n")
        logger.info(f"Model summary saved to {summary_path}")

@hydra.main(version_base=None, config_path="../configs", config_name="distillation_config")
def main(cfg: DictConfig):
    """Main function to run the distillation training."""
    import torch
    torch.cuda.is_available()

    # --- Setup Output Directory and Logging ---
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    # Configure file logging
    log_file = os.path.join(output_dir, "distillation_train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Hydra output directory: {output_dir}")
    logger.info("Full config:\n" + OmegaConf.to_yaml(cfg))
    
    # Construct absolute path for soft labels
    original_cwd = hydra.utils.get_original_cwd()
    soft_labels_path = os.path.join(original_cwd, cfg.dataset.soft_labels_path)

    if not os.path.exists(soft_labels_path):
        logger.error(f"Soft labels path does not exist: {soft_labels_path}")
        sys.exit(1)
        
    logger.info(f"Using soft labels from: {soft_labels_path}")
    
    # Initialize and run trainer
    try:
        trainer = DistillationTrainer(config=cfg, soft_labels_path=soft_labels_path, output_dir=output_dir)
        trainer.train()
        test_acc, report_dict = trainer.test()
        trainer.save_results(test_acc, report_dict)
    
        logger.info(f"--- Distillation Training Completed ---")
        logger.info(f"Final test accuracy: {test_acc:.4f}")
        logger.info(f"Results and plots saved to: {output_dir}")
    except Exception as e:
        logger.exception(f"An error occurred during training or testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 