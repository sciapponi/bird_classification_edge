#!/usr/bin/env python3
"""
Training script for knowledge distillation using BirdNET as teacher.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import math
from enum import Enum, auto

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from distillation.losses.distillation_loss import DistillationLoss, AdaptiveDistillationLoss
from distillation.losses.focal_loss import FocalLoss, FocalDistillationLoss, AdaptiveFocalDistillationLoss
from distillation.datasets.distillation_dataset import create_distillation_dataloader
from distillation.datasets.hybrid_dataset import create_hybrid_dataloader
from models import Improved_Phi_GRU_ATT
from distillation.optimizer.combined_optimizer import CombinedOptimizer
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
    Handles class filtering and soft label remapping automatically.
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
        
        # Create class mapping for soft label filtering
        self._create_class_mapping()
        
        logger.info(f"PreprocessedDistillationDataset initialized:")
        logger.info(f"  Preprocessed dataset samples: {len(self.preprocessed_dataset)}")
        logger.info(f"  Soft labels available: {len(self.soft_labels_data)}")
        logger.info(f"  Original soft labels classes: {self.original_num_classes}")
        logger.info(f"  Filtered soft labels classes: {self.filtered_num_classes}")
        if hasattr(self, 'class_mapping'):
            logger.info(f"  Class mapping: {dict(list(self.class_mapping.items())[:5])}..." if len(self.class_mapping) > 5 else f"  Class mapping: {self.class_mapping}")
    
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
    
    def _create_class_mapping(self):
        """Create mapping between original soft label classes and filtered dataset classes."""
        # Get original soft labels info
        self.original_num_classes = self.soft_labels_metadata.get('num_classes', 71)
        original_species = self.soft_labels_metadata.get('target_species', [])
        
        # Get filtered dataset classes
        if hasattr(self.preprocessed_dataset, 'get_classes'):
            filtered_classes = self.preprocessed_dataset.get_classes()
        else:
            # Fallback: use class names from class_to_idx
            if hasattr(self.preprocessed_dataset, 'class_to_idx'):
                filtered_classes = list(self.preprocessed_dataset.class_to_idx.keys())
            else:
                # Last resort: assume classes match indices
                filtered_classes = [f"Class_{i}" for i in range(len(self.preprocessed_dataset))]
        
        self.filtered_num_classes = len(filtered_classes)
        
        # Create mapping from original soft label indices to filtered indices
        self.class_mapping = {}
        self.filtered_classes = filtered_classes
        
        for filtered_idx, filtered_class in enumerate(filtered_classes):
            # Find corresponding index in original soft labels
            if filtered_class in original_species:
                original_idx = original_species.index(filtered_class)
                self.class_mapping[original_idx] = filtered_idx
            elif filtered_class in ['no_bird', 'non-bird'] and 'non-bird' in original_species:
                # Handle no_bird/non-bird class
                original_idx = original_species.index('non-bird')
                self.class_mapping[original_idx] = filtered_idx
        
        logger.info(f"Created soft label mapping: {len(self.class_mapping)} classes mapped")
    
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
        # Note: For filtered datasets, class_idx should already be in the correct range [0, num_filtered_classes-1]
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
        Get filtered soft label for a given filename.
        
        Args:
            filename: Original filename
            
        Returns:
            torch.Tensor: Filtered soft label probabilities
        """
        # Remove extension and path to match soft labels keys
        base_filename = Path(filename).stem
        
        # Try to find soft label by exact match first
        original_soft_label = None
        if base_filename in self.soft_labels_data:
            original_soft_label = self.soft_labels_data[base_filename]
        else:
            # Try partial matching (filename might contain additional info)
            matching_keys = [key for key in self.soft_labels_data.keys() if base_filename in key or key in base_filename]
            if matching_keys:
                # Use first match
                original_soft_label = self.soft_labels_data[matching_keys[0]]
                logger.debug(f"Soft label found using partial match: {base_filename} -> {matching_keys[0]}")
        
        if original_soft_label is not None:
            # Filter and remap the soft labels
            filtered_soft_label = self._filter_soft_labels(original_soft_label)
        else:
            # Create uniform distribution for filtered classes
            filtered_soft_label = [1.0 / self.filtered_num_classes] * self.filtered_num_classes
            # Only log first 10 warnings to avoid spam
            if not hasattr(self, '_warning_count'):
                self._warning_count = 0
            if self._warning_count < 10:
                logger.warning(f"No soft label found for {base_filename}, using uniform distribution with {self.filtered_num_classes} classes")
                self._warning_count += 1
            elif self._warning_count == 10:
                logger.warning(f"... suppressing further 'No soft label found' warnings (many files without soft labels detected)")
                self._warning_count += 1
        
        return torch.tensor(filtered_soft_label, dtype=torch.float32)
    
    def _filter_soft_labels(self, original_soft_labels):
        """
        Filter and remap original soft labels to filtered classes.
        
        Args:
            original_soft_labels: List of probabilities for original classes
            
        Returns:
            List of probabilities for filtered classes
        """
        if not hasattr(self, 'class_mapping'):
            # No filtering needed
            return original_soft_labels
        
        # Create filtered soft labels
        filtered_soft_labels = [0.0] * self.filtered_num_classes
        
        # Map probabilities from original to filtered classes
        for original_idx, filtered_idx in self.class_mapping.items():
            if original_idx < len(original_soft_labels):
                filtered_soft_labels[filtered_idx] += original_soft_labels[original_idx]
        
        # Normalize to ensure sum = 1
        total_prob = sum(filtered_soft_labels)
        if total_prob > 0:
            filtered_soft_labels = [prob / total_prob for prob in filtered_soft_labels]
        else:
            # Fallback to uniform distribution
            filtered_soft_labels = [1.0 / self.filtered_num_classes] * self.filtered_num_classes
        
        return filtered_soft_labels
    
    def get_soft_labels_info(self):
        """Get information about filtered soft labels"""
        return {
            'total_files_with_soft_labels': len(self.soft_labels_data),
            'num_classes': self.filtered_num_classes,
            'target_species': self.filtered_classes,
            'confidence_threshold': self.soft_labels_metadata.get('confidence_threshold'),
            'files_processed': self.soft_labels_metadata.get('total_files_processed')
        }
    
    def get_classes(self):
        """Get list of filtered class names"""
        return self.filtered_classes
    
    def get_original_classes(self):
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

# Define the phases for alternating optimization
class OptimizationPhase(Enum):
    JOINT = auto()
    MAIN_ONLY = auto()
    FILTER_ONLY = auto()

class DistillationTrainer:
    def __init__(self, config, soft_labels_path, output_dir):
        self.config = config
        self.soft_labels_path = soft_labels_path
        self.output_dir = Path(output_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components - THIS IS THE CORRECT STRUCTURE
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
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
        self.learning_rates = []
        self.class_names = []
        
        # Filter parameters history (for combined_log_linear)
        self.breakpoint_history = []
        self.transition_width_history = []
        self.filter_params_history = []
        
        logger.info(f"Initialized trainer on device: {self.device}")
        logger.info(f"Outputs will be saved to: {self.output_dir}")
        
    def _get_optimization_phase(self, current_epoch: int) -> OptimizationPhase:
        """
        Determines the optimization phase for the current epoch based on the configuration.
        """
        alt_opt_config = self.config.get('alternating_optimization', {})
        if not alt_opt_config.get('enabled', False):
            return OptimizationPhase.JOINT

        initial_joint = alt_opt_config.get('initial_joint_epochs', 0)
        final_joint = alt_opt_config.get('final_joint_epochs', 0)
        total_epochs = self.config.training.epochs

        # Phase 1: Initial joint training
        if current_epoch <= initial_joint:
            return OptimizationPhase.JOINT

        # Phase 3: Final joint training
        if current_epoch > total_epochs - final_joint:
            return OptimizationPhase.JOINT

        # Phase 2: Alternating optimization
        main_opt_epochs = alt_opt_config.get('main_opt_epochs', 1)
        filter_opt_epochs = alt_opt_config.get('filter_opt_epochs', 1)
        cycle_len = main_opt_epochs + filter_opt_epochs
        
        # This should not happen if configured correctly, but as a safeguard:
        if cycle_len == 0:
            return OptimizationPhase.JOINT
            
        # Calculate position within the alternating block
        epoch_in_alt_phase = current_epoch - initial_joint
        position_in_cycle = (epoch_in_alt_phase - 1) % cycle_len

        if position_in_cycle < main_opt_epochs:
            return OptimizationPhase.MAIN_ONLY
        else:
            return OptimizationPhase.FILTER_ONLY
            
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
        
        # Get base learning rate
        base_lr = self.config.optimizer.get('lr', 0.001)
        
        # Check for separate multipliers (new system)
        breakpoint_lr_mult = self.config.get('breakpoint_lr_multiplier', None)
        transition_width_lr_mult = self.config.get('transition_width_lr_multiplier', None)
        
        # Fallback to unified multiplier (old system for backward compatibility)
        filter_lr_mult = self.config.get('filter_lr_multiplier', 1.0)
        
        # Use separate multipliers if available, otherwise use unified multiplier
        if breakpoint_lr_mult is None:
            breakpoint_lr_mult = filter_lr_mult
        if transition_width_lr_mult is None:
            transition_width_lr_mult = filter_lr_mult
        
        # Check if we need different learning rates for filter parameters
        if breakpoint_lr_mult != 1.0 or transition_width_lr_mult != 1.0:
            # Separate filter parameters into breakpoint, transition_width, and other filter params
            breakpoint_params = []
            transition_width_params = []
            other_filter_params = []
            other_params = []
            
            for name, param in self.model.named_parameters():
                if 'breakpoint' in name:
                    breakpoint_params.append(param)
                    logger.info(f"Breakpoint parameter: {name}, shape: {param.shape}")
                elif 'transition_width' in name:
                    transition_width_params.append(param)
                    logger.info(f"Transition width parameter: {name}, shape: {param.shape}")
                elif 'filter_bank' in name:
                    other_filter_params.append(param)
                    logger.info(f"Other filter parameter: {name}, shape: {param.shape}")
                else:
                    other_params.append(param)
            
            # Create parameter groups with different learning rates
            param_groups = []
            
            # Base parameters (non-filter)
            if other_params:
                param_groups.append({'params': other_params, 'lr': base_lr})
            
            # Breakpoint parameters
            if breakpoint_params:
                breakpoint_lr = base_lr * breakpoint_lr_mult
                param_groups.append({'params': breakpoint_params, 'lr': breakpoint_lr})
                logger.info(f"Breakpoint learning rate: {base_lr} × {breakpoint_lr_mult} = {breakpoint_lr}")
            
            # Transition width parameters  
            if transition_width_params:
                transition_width_lr = base_lr * transition_width_lr_mult
                param_groups.append({'params': transition_width_params, 'lr': transition_width_lr})
                logger.info(f"Transition width learning rate: {base_lr} × {transition_width_lr_mult} = {transition_width_lr}")
            
            # Other filter parameters (like filter_bank for fully learnable)
            if other_filter_params:
                other_filter_lr = base_lr * filter_lr_mult  # Use unified multiplier for other filter params
                param_groups.append({'params': other_filter_params, 'lr': other_filter_lr})
                logger.info(f"Other filter parameters learning rate: {base_lr} × {filter_lr_mult} = {other_filter_lr}")
            
            # Create optimizer with parameter groups instead of using hydra
            optimizer_class = getattr(torch.optim, self.config.optimizer._target_.split('.')[-1])
            optimizer_kwargs = {k: v for k, v in self.config.optimizer.items() if k != '_target_'}
            optimizer_kwargs.pop('lr', None)  # Remove base lr since we're using param groups
            
            self.optimizer = optimizer_class(param_groups, **optimizer_kwargs)
            
            logger.info(f"Using separate learning rates:")
            logger.info(f"  Base parameters: {base_lr}")
            if breakpoint_params:
                logger.info(f"  Breakpoint: {breakpoint_lr} (multiplier: {breakpoint_lr_mult})")
            if transition_width_params:
                logger.info(f"  Transition width: {transition_width_lr} (multiplier: {transition_width_lr_mult})")
            if other_filter_params:
                logger.info(f"  Other filters: {other_filter_lr} (multiplier: {filter_lr_mult})")
        else:
            # Standard optimizer setup
            self.optimizer = hydra.utils.instantiate(
                self.config.optimizer,
                params=self.model.parameters()
            )
            logger.info(f"Using single learning rate: {self.config.optimizer.get('lr', 'default')}")
        
        # Setup schedulers
        self.scheduler = None
        self.filter_scheduler = None
        
        # Check if we have separate schedulers for filter parameters
        if (breakpoint_lr_mult != 1.0 or transition_width_lr_mult != 1.0) and 'filter_scheduler' in self.config:
            # Create separate optimizers for main and filter parameters
            main_params = []
            filter_params = []
            
            for name, param in self.model.named_parameters():
                if any(filter_name in name for filter_name in ['breakpoint', 'transition_width', 'filter_bank']):
                    filter_params.append(param)
                else:
                    main_params.append(param)
            
            if main_params and filter_params:
                # Create main optimizer for non-filter parameters
                optimizer_kwargs_main = {k: v for k, v in self.config.optimizer.items() if k != '_target_'}
                self.main_optimizer = optimizer_class(main_params, **optimizer_kwargs_main)
                
                # Create filter optimizer for filter parameters  
                optimizer_kwargs_filter = {k: v for k, v in self.config.optimizer.items() if k != '_target_'}
                filter_lr = base_lr * max(breakpoint_lr_mult, transition_width_lr_mult)  # Use higher LR
                optimizer_kwargs_filter['lr'] = filter_lr
                self.filter_optimizer = optimizer_class(filter_params, **optimizer_kwargs_filter)
                
                # Setup separate schedulers
                if 'scheduler' in self.config:
                    self.scheduler = hydra.utils.instantiate(
                        self.config.scheduler,
                        optimizer=self.main_optimizer
                    )
                    
                if 'filter_scheduler' in self.config and self.config.filter_scheduler is not None:
                    self.filter_scheduler = hydra.utils.instantiate(
                        self.config.filter_scheduler,
                        optimizer=self.filter_optimizer
                    )
                
                logger.info(f"Using separate optimizers and schedulers:")
                logger.info(f"  Main optimizer: {type(self.main_optimizer).__name__}")
                logger.info(f"  Filter optimizer: {type(self.filter_optimizer).__name__}")
                if self.scheduler:
                    logger.info(f"  Main scheduler: {type(self.scheduler).__name__}")
                if self.filter_scheduler:
                    logger.info(f"  Filter scheduler: {type(self.filter_scheduler).__name__}")
                else:
                    logger.info(f"  Filter scheduler: None (constant LR)")
                    
                # Override self.optimizer to be a combined object for backward compatibility
                self.optimizer = CombinedOptimizer(self.main_optimizer, self.filter_optimizer)
                
                # NUOVO: Setup filter exploration se specificato nel config
                self._setup_filter_exploration()
                
                # NUOVO: Setup alternating optimization
                self._setup_alternating_optimization()
                
            else:
                # Fallback al sistema precedente
                logger.warning("Could not separate main and filter parameters, using single optimizer")
                # Standard scheduler setup
                if 'scheduler' in self.config:
                    self.scheduler = hydra.utils.instantiate(
                        self.config.scheduler,
                        optimizer=self.optimizer
                    )
        else:
            # Standard scheduler setup
            if 'scheduler' in self.config:
                self.scheduler = hydra.utils.instantiate(
                    self.config.scheduler,
                    optimizer=self.optimizer
                )
        
        logger.info(f"Optimizer: {type(self.optimizer).__name__}")
        if hasattr(self, 'main_optimizer'):
            logger.info(f"Main Optimizer: {type(self.main_optimizer).__name__}")
            logger.info(f"Filter Optimizer: {type(self.filter_optimizer).__name__}")
        if self.scheduler:
            logger.info(f"Main Scheduler: {type(self.scheduler).__name__}")
        if self.filter_scheduler:
            logger.info(f"Filter Scheduler: {type(self.filter_scheduler).__name__}")
    
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

    def _compute_automatic_class_weights(self, train_dataset):
        """Compute automatic class weights based on training data distribution"""
        import torch
        from collections import Counter
        
        logger.info("Analyzing training data for automatic class weights...")
        
        # Count class frequencies in training data
        class_counts = Counter()
        
        # Sample subset of training data to compute weights (faster)
        sample_size = self.config.loss.get('auto_weights_sample_size', 5000)
        sample_size = min(len(train_dataset), sample_size)  # Ensure we don't sample more than available
        indices = torch.randperm(len(train_dataset))[:sample_size]
        
        logger.info(f"Sampling {sample_size} examples from {len(train_dataset)} total training samples...")
        
        for idx in tqdm(indices, desc="Computing class weights"):
            if hasattr(train_dataset, 'dataset'):
                # Handle DistillationDataset wrapper
                _, hard_label, _ = train_dataset.dataset[idx]
            else:
                # Handle direct dataset
                if len(train_dataset[idx]) >= 2:
                    _, hard_label = train_dataset[idx][:2]
                else:
                    continue
                    
            if isinstance(hard_label, torch.Tensor):
                hard_label = hard_label.item()
            class_counts[hard_label] += 1
        
        logger.info(f"Class distribution in sample: {dict(class_counts)}")
        
        # Calculate inverse frequency weights
        total_samples = sum(class_counts.values())
        num_classes = self.num_classes # ALWAYS use the official number of classes
        
        # Compute weights: weight = total_samples / (num_classes * class_count)
        class_weights = []
        for class_idx in range(num_classes):
            if class_idx in class_counts:
                weight = total_samples / (num_classes * class_counts[class_idx])
            else:
                weight = 1.0  # Default weight for missing classes
            class_weights.append(weight)
        
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        logger.info(f"Computed class weights: {class_weights}")
        logger.info(f"Weight range: {class_weights_tensor.min().item():.3f} - {class_weights_tensor.max().item():.3f}")
        
        # Create criterion with computed weights
        if hasattr(self, '_setup_focal_distillation_with_auto_weights') and self._setup_focal_distillation_with_auto_weights:
            # Focal distillation with automatic weights
            params = self._focal_distillation_params
            if params['adaptive']:
                self.criterion = AdaptiveFocalDistillationLoss(
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    temperature=params['temperature'],
                    class_weights=class_weights_tensor,
                    adaptation_rate=params['adaptation_rate']
                )
                logger.info("Created AdaptiveFocalDistillationLoss with automatic class weights")
            else:
                self.criterion = FocalDistillationLoss(
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    temperature=params['temperature'],
                    class_weights=class_weights_tensor
                )
                logger.info("Created FocalDistillationLoss with automatic class weights")
        elif hasattr(self, '_setup_focal_with_auto_weights') and self._setup_focal_with_auto_weights:
            # Pure focal loss with automatic weights
            self.criterion = FocalLoss(
                alpha=class_weights_tensor,
                gamma=self.config.loss.get('gamma', 2.0)
            )
            logger.info("Created FocalLoss with automatic class weights")
        else:
            # Fallback
            logger.warning("Unknown automatic weights setup, using AdaptiveFocalDistillationLoss")
            self.criterion = AdaptiveFocalDistillationLoss(
                alpha=self.config.distillation.get('alpha', 0.4),
                gamma=self.config.loss.get('gamma', 4.0),
                temperature=self.config.distillation.get('temperature', 3.0),
                class_weights=class_weights_tensor,
                adaptation_rate=self.config.distillation.get('adaptation_rate', 0.1)
            )

    def train_epoch(self, phase):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {self.epoch + 1}')
        
        for batch_idx, (inputs, hard_targets, soft_targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            hard_targets = hard_targets.to(self.device)
            
            # Handle soft targets (can be None)
            if soft_targets is not None:
                soft_targets = soft_targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            if isinstance(self.criterion, (DistillationLoss, FocalDistillationLoss, AdaptiveDistillationLoss, AdaptiveFocalDistillationLoss)):
                if soft_targets is not None:
                    loss, hard_loss, soft_loss = self.criterion(outputs, hard_targets, soft_targets)
                else:
                    # Fallback to hard loss only
                    loss = F.cross_entropy(outputs, hard_targets)
                    hard_loss = loss
                    soft_loss = torch.tensor(0.0)
            else:
                # Pure focal loss or other loss
                loss = self.criterion(outputs, hard_targets)
                hard_loss = loss
                soft_loss = torch.tensor(0.0)

            # Add regularization loss for filters
            if hasattr(self, 'filter_regularization') and any(self.filter_regularization.values()):
                reg_loss = self._compute_filter_regularization_loss()
                loss += reg_loss
            
            # Backward pass
            loss.backward()

            # Apply filter exploration techniques before optimizer step
            if hasattr(self, 'filter_exploration') and any(self.filter_exploration.values()):
                self._apply_filter_exploration(self.epoch)

            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_hard_loss += hard_loss.item()
            total_soft_loss += soft_loss.item()
            
            _, predicted = outputs.max(1)
            total += hard_targets.size(0)
            correct += predicted.eq(hard_targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.1f}%'
            })
        
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_hard_loss = total_hard_loss / len(self.train_loader)
        avg_soft_loss = total_soft_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        # Store detailed losses for logging
        self.hard_losses.append(avg_hard_loss)
        self.soft_losses.append(avg_soft_loss)
        
        return avg_loss, accuracy

    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, hard_targets, soft_targets in self.val_loader:
                inputs = inputs.to(self.device)
                hard_targets = hard_targets.to(self.device)
                
                # Handle soft targets (can be None)
                if soft_targets is not None:
                    soft_targets = soft_targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                if isinstance(self.criterion, (DistillationLoss, FocalDistillationLoss, AdaptiveDistillationLoss, AdaptiveFocalDistillationLoss)):
                    if soft_targets is not None:
                        loss, hard_loss, soft_loss = self.criterion(outputs, hard_targets, soft_targets)
                    else:
                        # Fallback to hard loss only
                        loss = F.cross_entropy(outputs, hard_targets)
                        hard_loss = loss
                        soft_loss = torch.tensor(0.0)
                else:
                    # Pure focal loss or other loss
                    loss = self.criterion(outputs, hard_targets)
                    hard_loss = loss
                    soft_loss = torch.tensor(0.0)
                
                # Statistics
                total_loss += loss.item()
                total_hard_loss += hard_loss.item()
                total_soft_loss += soft_loss.item()
                
                _, predicted = outputs.max(1)
                total += hard_targets.size(0)
                correct += predicted.eq(hard_targets).sum().item()
        
        # Calculate averages
        avg_loss = total_loss / len(self.val_loader)
        avg_hard_loss = total_hard_loss / len(self.val_loader)
        avg_soft_loss = total_soft_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def train(self):
        """Main training loop"""
        # Setup components before training
        self.train_dataset, self.val_dataset, self.test_dataset = self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        self.setup_criterion()
        
        # If criterion is None, we need to compute automatic class weights
        if self.criterion is None:
            logger.info("Computing automatic class weights from training data...")
            self._compute_automatic_class_weights(self.train_dataset)
        
        # Early stopping
        early_stopper = EarlyStopping(
            patience=self.config.training.get('patience', 10),
            min_delta=self.config.training.get('min_delta', 0.001)
        )
        
        logger.info("Starting training loop...")
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Track learning rate at the start of the epoch
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Optional pause between training and validation to reduce CPU stress
            validation_pause = self.config.training.get('validation_pause', 0)
            if validation_pause > 0:
                logger.info(f"Pausing {validation_pause} seconds between training and validation to reduce CPU load...")
                time.sleep(validation_pause)
            
            # Determine the current optimization phase
            current_phase = self._get_optimization_phase(epoch)
            logger.info(f"Epoch {epoch + 1}: Starting phase '{current_phase.name}'")

            # Set the optimizers for the current phase
            if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'set_active_optimizers'):
                if current_phase == OptimizationPhase.MAIN_ONLY:
                    self.optimizer.set_active_optimizers(['main'])
                    self._unfreeze_main_network() # Ensure main is trainable
                elif current_phase == OptimizationPhase.FILTER_ONLY:
                    self.optimizer.set_active_optimizers(['filter'])
                    self._freeze_main_network() # Freeze main for filter training
                elif current_phase == OptimizationPhase.JOINT:
                    self.optimizer.set_active_optimizers(['main', 'filter'])
                    self._unfreeze_main_network() # Ensure all are trainable

            # Train for one epoch
            train_loss, train_acc = self.train_epoch(current_phase)
            
            # Validate and get metrics
            val_loss, val_acc = self.validate_epoch()

            # Store training history
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Track filter parameters
            if hasattr(self, '_log_filter_parameters'):
                self._log_filter_parameters()
            
            # Log epoch results
            epoch_duration = time.time() - epoch_start_time
            self.log_epoch_results(epoch, train_loss, train_acc, val_loss, val_acc, epoch_duration)
            
            # Monitor fully learnable filters if enabled
            if (self.config.get('logging', {}).get('monitor_filters', False) and 
                self.config.model.get('spectrogram_type') == 'fully_learnable'):
                self._log_fully_learnable_filter_stats()
                
                # Save filter visualization periodically
                viz_interval = self.config.get('logging', {}).get('filter_visualization_interval', 10)
                if viz_interval > 0 and self.epoch % viz_interval == 0:
                    self._save_filter_visualization()
            
            # Step the schedulers
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # ... and for the filter scheduler if it exists
            if self.filter_scheduler:
                if isinstance(self.filter_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.filter_scheduler.step(val_loss)
                else:
                    self.filter_scheduler.step()
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}, Acc: {self.best_val_acc:.4f}")
                save_checkpoint(self.model, self.optimizer, self.epoch, self.best_val_loss, self.best_val_acc, self.output_dir / 'best_model.pt')
            
            # Early stopping check
            if early_stopper(val_loss):
                logger.info("Early stopping triggered.")
                break
        
            # NUOVO: Spostato il reset del momentum qui per eseguirlo una volta per epoca
            if (hasattr(self, 'filter_exploration') and
                self.filter_exploration['enable_momentum_reset'] and 
                self.epoch > 0 and 
                self.epoch % self.filter_exploration['momentum_reset_period'] == 0):
                self._reset_filter_momentum()
        
        logger.info("Training loop finished.")
        self.save_training_plots()
    
    def log_epoch_results(self, epoch, train_loss, train_acc, val_loss, val_acc, epoch_duration):
        """Logs the results of a training epoch."""
        log_message = (
            f"Epoch {epoch + 1}/{self.config.training.epochs} | "
            f"Duration: {epoch_duration:.2f}s | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )
        if self.learning_rates:
            log_message += f" | LR: {self.learning_rates[-1]:.6f}"
        
        logger.info(log_message)

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
            logger.info(f"Epoch {self.epoch + 1} - Filter params: Breakpoint={current_breakpoint:.2f}Hz, Transition Width={current_transition_width:.2f}")
    
    def track_filter_parameters(self, epoch):
        """Alias for _log_filter_parameters for backward compatibility."""
        self._log_filter_parameters()
    
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
        log_msg = f"Epoch {self.epoch + 1} - Filter Bank Stats: "
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
            fig.suptitle(f'Fully Learnable Filter Bank - Epoch {self.epoch + 1}', fontsize=16)
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
                fig.suptitle(f'Filter Evolution Over Training (Epochs 1-{self.epoch + 1})', fontsize=16)
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
            ax1.set_title(f'All 64 Filters - Epoch {self.epoch + 1}')
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
            
            logger.info(f"Enhanced filter visualizations saved for epoch {self.epoch + 1}")
            
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

        # --- EMERGENCY DEBUG FOR CLASS NAMES ISSUE ---
        logger.info("🚨 EMERGENCY DEBUG: Investigating class_names issue")
        logger.info(f"self.class_names = {getattr(self, 'class_names', 'ATTRIBUTE_NOT_FOUND')}")
        logger.info(f"Type of self.class_names = {type(getattr(self, 'class_names', None))}")
        logger.info(f"Length of self.class_names = {len(getattr(self, 'class_names', []))}")
        logger.info(f"self.num_classes = {getattr(self, 'num_classes', 'ATTRIBUTE_NOT_FOUND')}")
        
        # Get config info for debugging
        if hasattr(self, 'config'):
            allowed_classes = self.config.dataset.get('allowed_bird_classes', 'NOT_FOUND')
            use_no_bird = self.config.dataset.get('use_no_bird_class', 'NOT_FOUND')
            logger.info(f"Config allowed_bird_classes = {allowed_classes}")
            logger.info(f"Config use_no_bird_class = {use_no_bird}")
        
        # Check what we can get from test_loader
        if hasattr(self, 'test_loader') and hasattr(self.test_loader, 'dataset'):
            dataset = self.test_loader.dataset
            logger.info(f"Test dataset type: {type(dataset)}")
            if hasattr(dataset, 'get_classes'):
                try:
                    dataset_classes = dataset.get_classes()
                    logger.info(f"Dataset.get_classes() = {dataset_classes}")
                except Exception as e:
                    logger.info(f"Error calling dataset.get_classes(): {e}")
            
            if hasattr(dataset, 'class_names'):
                logger.info(f"Dataset.class_names = {dataset.class_names}")
            if hasattr(dataset, 'filtered_classes'):
                logger.info(f"Dataset.filtered_classes = {dataset.filtered_classes}")

        # --- EMERGENCY FIX ---
        class_names = getattr(self, 'class_names', [])
        
        if not class_names or len(class_names) == 0:
            logger.error("🚨 CRITICAL: class_names is empty! Applying emergency fix...")
            
            # Try to get from config directly
            if hasattr(self, 'config'):
                emergency_classes = self.config.dataset.get('allowed_bird_classes', [])
                if emergency_classes:
                    class_names = list(emergency_classes)
                    if self.config.dataset.get('use_no_bird_class', False):
                        class_names.append('No Bird')
                    logger.error(f"🔧 Emergency fix from config: {class_names}")
                else:
                    # Ultimate fallback
                    unique_labels = sorted(list(set(all_true + all_preds)))
                    class_names = [f"Class_{i}" for i in range(max(unique_labels) + 1)]
                    logger.error(f"🔧 Emergency fix with generic names: {class_names}")
            else:
                # Ultimate fallback without config
                unique_labels = sorted(list(set(all_true + all_preds)))
                class_names = [f"Class_{i}" for i in range(max(unique_labels) + 1)]
                logger.error(f"🔧 Emergency fix with generic names (no config): {class_names}")
        
        logger.info(f"Final class_names for testing: {class_names}")
        logger.info("🚨 END EMERGENCY DEBUG")

        # --- NEW, SIMPLIFIED LOGIC ---
        # Get the full list of class names directly from the trainer.
        # This is now robustly populated by setup_data() based on the experiment config.
        # class_names = self.class_names  # REMOVED - using the emergency-fixed version above
        
        # Generate classification report using all classes from the config
        report_str = classification_report(
            all_true, all_preds, 
            labels=list(range(len(class_names))), # Ensure all classes are included
            target_names=class_names, 
            zero_division=0
        )
        report_dict = classification_report(
            all_true, all_preds, 
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True, 
            zero_division=0
        )
        
        logger.info(f"Test Report:\n{report_str}")

        # Save confusion matrix
        png_path = self.output_dir / 'confusion_matrix.png'
        csv_path = self.output_dir / 'confusion_matrix.csv'
        save_confusion_matrix(all_true, all_preds, class_names, png_path, csv_path)

        # Save results
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

    def _setup_filter_exploration(self):
        """Setup advanced filter parameter exploration"""
        # Get exploration config
        exploration_config = self.config.get('filter_exploration', {})
        regularization_config = self.config.get('filter_regularization', {})
        
        # Store exploration parameters
        self.filter_exploration = {
            'enable_noise': exploration_config.get('enable_noise', False),
            'noise_std': exploration_config.get('noise_std', 0.1),
            'enable_oscillation': exploration_config.get('enable_oscillation', False),
            'oscillation_amplitude': exploration_config.get('oscillation_amplitude', 50.0),
            'oscillation_period': exploration_config.get('oscillation_period', 10),
            'enable_momentum_reset': exploration_config.get('enable_momentum_reset', False),
            'momentum_reset_period': exploration_config.get('momentum_reset_period', 20),
        }
        
        self.filter_regularization = {
            'enable_range_penalty': regularization_config.get('enable_range_penalty', False),
            'sensible_breakpoint_min': regularization_config.get('sensible_breakpoint_min', 50.0),
            'sensible_breakpoint_max': regularization_config.get('sensible_breakpoint_max', 8000.0),
            'range_penalty_weight': regularization_config.get('range_penalty_weight', 0.001),
            'enable_diversity_bonus': regularization_config.get('enable_diversity_bonus', False),
            'diversity_bonus_weight': regularization_config.get('diversity_bonus_weight', 0.0005),
            'enable_stability_bonus': regularization_config.get('enable_stability_bonus', False),
            'stability_bonus_weight': regularization_config.get('stability_bonus_weight', 0.01),
            'stability_window': regularization_config.get('stability_window', 5),
        }
        
        # Initialize exploration state
        self.exploration_state = {
            'last_momentum_reset': 0,
            'breakpoint_history': [],
            'transition_width_history': [],
            'exploration_direction': 1.0  # 1.0 or -1.0 for oscillation
        }
        
        if any(self.filter_exploration.values()) or any(self.filter_regularization.values()):
            logger.info("🔬 Advanced filter exploration enabled:")
            if self.filter_exploration['enable_noise']:
                logger.info(f"  ✅ Gradient noise: std={self.filter_exploration['noise_std']}")
            if self.filter_exploration['enable_oscillation']:
                logger.info(f"  ✅ Oscillation: amplitude={self.filter_exploration['oscillation_amplitude']}Hz, period={self.filter_exploration['oscillation_period']} epochs")
            if self.filter_exploration['enable_momentum_reset']:
                logger.info(f"  ✅ Momentum reset: every {self.filter_exploration['momentum_reset_period']} epochs")
            if self.filter_regularization['enable_range_penalty']:
                logger.info(f"  ✅ Range penalty: [{self.filter_regularization['sensible_breakpoint_min']}, {self.filter_regularization['sensible_breakpoint_max']}] Hz")
            if self.filter_regularization['enable_diversity_bonus']:
                logger.info(f"  ✅ Diversity bonus: weight={self.filter_regularization['diversity_bonus_weight']}")
            if self.filter_regularization.get('enable_stability_bonus', False):
                logger.info(f"  ✅ Stability bonus: window={self.filter_regularization.get('stability_window', 5)}, weight={self.filter_regularization.get('stability_bonus_weight', 0.01)}")
    
    def _reset_filter_momentum(self):
        """Resets the momentum of the filter optimizer."""
        if hasattr(self, 'filter_optimizer'):
            # Reset momentum in AdamW
            for group in self.filter_optimizer.param_groups:
                for p in group['params']:
                    if p in self.filter_optimizer.state:
                        state = self.filter_optimizer.state[p]
                        state['exp_avg'] = torch.zeros_like(p)  # Reset momentum
                        state['exp_avg_sq'] = torch.zeros_like(p)  # Reset second moment
                        
            logger.info(f"🔄 Reset filter optimizer momentum at epoch {self.epoch + 1}")
            self.exploration_state['last_momentum_reset'] = self.epoch

    def _apply_filter_exploration(self, epoch):
        """Apply exploration techniques to filter parameters"""
        if not hasattr(self, 'filter_exploration'):
            return
            
        # Get filter parameters
        filter_params = []
        if hasattr(self.model, 'combined_log_linear_spec'):
            if hasattr(self.model.combined_log_linear_spec, 'breakpoint'):
                filter_params.append(('breakpoint', self.model.combined_log_linear_spec.breakpoint))
            if hasattr(self.model.combined_log_linear_spec, 'transition_width'):
                filter_params.append(('transition_width', self.model.combined_log_linear_spec.transition_width))
        
        if not filter_params:
            return
            
        # 1. Add gradient noise
        if self.filter_exploration['enable_noise']:
            for name, param in filter_params:
                if param.grad is not None:
                    noise_scale = self.filter_exploration['noise_std']
                    if name == 'transition_width':
                        noise_scale *= 0.1  # Smaller noise for transition_width
                    
                    noise = torch.randn_like(param.grad) * noise_scale
                    param.grad.add_(noise)
        
        # 2. Add oscillation force
        if self.filter_exploration['enable_oscillation']:
            period = self.filter_exploration['oscillation_period']
            amplitude = self.filter_exploration['oscillation_amplitude']
            
            # Calculate oscillation phase
            phase = (epoch % period) / period * 2 * math.pi
            oscillation_force = amplitude * math.sin(phase)
            
            for name, param in filter_params:
                if name == 'breakpoint' and param.grad is not None:
                    # Add oscillation as additional gradient component
                    param.grad.add_(torch.tensor(oscillation_force / 1000.0, device=param.device))  # Scale down
        
        # 3. Momentum reset - RIMOSSO DA QUI
    
    def _compute_filter_regularization_loss(self):
        """Compute additional loss terms for filter regularization"""
        if not hasattr(self, 'filter_regularization'):
            return torch.tensor(0.0, device=self.device)
            
        total_reg_loss = torch.tensor(0.0, device=self.device)
        
        if not hasattr(self.model, 'combined_log_linear_spec'):
            return total_reg_loss
            
        breakpoint = self.model.combined_log_linear_spec.breakpoint
        transition_width = self.model.combined_log_linear_spec.transition_width
        
        # Traccia la storia dei parametri per i bonus/penalità
        self.exploration_state['breakpoint_history'].append(breakpoint.item())
        self.exploration_state['transition_width_history'].append(transition_width.item())
        
        # Mantieni solo la storia recente (es. ultime 20 epoche per evitare un accumulo eccessivo)
        max_history = 20
        if len(self.exploration_state['breakpoint_history']) > max_history:
            self.exploration_state['breakpoint_history'] = self.exploration_state['breakpoint_history'][-max_history:]
            self.exploration_state['transition_width_history'] = self.exploration_state['transition_width_history'][-max_history:]
        
        # 1. Range penalty - penalize values outside sensible range
        if self.filter_regularization['enable_range_penalty']:
            min_val = self.filter_regularization['sensible_breakpoint_min']
            max_val = self.filter_regularization['sensible_breakpoint_max']
            weight = self.filter_regularization['range_penalty_weight']
            
            # Soft penalty using smooth approximation
            penalty = torch.relu(min_val - breakpoint) + torch.relu(breakpoint - max_val)
            total_reg_loss += weight * penalty
        
        # 2. Diversity bonus - encourage exploration of different values
        if self.filter_regularization['enable_diversity_bonus']:
            weight = self.filter_regularization['diversity_bonus_weight']
            
            # Bonus per essere differente dai valori recenti
            # Usa una finestra più piccola per la diversità, es. 5 epoche
            diversity_window = 5
            if len(self.exploration_state['breakpoint_history']) > diversity_window:
                recent_values = torch.tensor(self.exploration_state['breakpoint_history'][-diversity_window:], device=self.device)
                diversity = torch.std(recent_values)  # Higher std = more diversity = bonus
                total_reg_loss -= weight * diversity  # Negative loss = bonus
        
        # 3. Stability bonus (penalità per l'instabilità)
        if self.filter_regularization.get('enable_stability_bonus', False):
            weight = self.filter_regularization.get('stability_bonus_weight', 0.01)
            window = self.filter_regularization.get('stability_window', 5)
            
            if len(self.exploration_state['breakpoint_history']) >= window:
                # Penalità basata sulla deviazione standard dei parametri recenti
                recent_bps = torch.tensor(self.exploration_state['breakpoint_history'][-window:], device=self.device)
                recent_tws = torch.tensor(self.exploration_state['transition_width_history'][-window:], device=self.device)
                
                instability_penalty = torch.std(recent_bps) + torch.std(recent_tws) * 0.1 # Pesa meno la TW
                total_reg_loss += weight * instability_penalty

        return total_reg_loss
    
    def _setup_alternating_optimization(self):
        """Setup alternating optimization system"""
        alternating_config = self.config.get('alternating_optimization', {})
        
        self.alternating_opt = {
            'enable': alternating_config.get('enable', False),
            'main_epochs': alternating_config.get('main_epochs', 3),
            'filter_epochs': alternating_config.get('filter_epochs', 2),
            'validation_feedback': alternating_config.get('validation_feedback', True),
            'filter_candidates_test': alternating_config.get('filter_candidates_test', 5),
            'search_range_factor': alternating_config.get('search_range_factor', 0.3),
            'enable_joint_phases': alternating_config.get('enable_joint_phases', False),
            'joint_phase_frequency': alternating_config.get('joint_phase_frequency', 15),
            'joint_phase_duration': alternating_config.get('joint_phase_duration', 3),
            'current_phase': 'main',  # 'main', 'filter', or 'joint'
            'phase_counter': 0,
            'global_epoch_counter': 0,
            'filter_history': []  # Track filter performance
        }
        
        if self.alternating_opt['enable']:
            logger.info("🔄 Advanced Alternating Optimization enabled:")
            logger.info(f"  📊 Main network epochs: {self.alternating_opt['main_epochs']}")
            logger.info(f"  🎛️ Filter epochs: {self.alternating_opt['filter_epochs']}")
            logger.info(f"  ✅ Validation feedback: {self.alternating_opt['validation_feedback']}")
            logger.info(f"  🔍 Filter candidates per test: {self.alternating_opt['filter_candidates_test']}")
            if self.alternating_opt['enable_joint_phases']:
                logger.info(f"  🤝 Joint phases: every {self.alternating_opt['joint_phase_frequency']} epochs for {self.alternating_opt['joint_phase_duration']} epochs")
    
    def _should_optimize_main_network(self, epoch):
        """Determine if we should optimize main network this epoch"""
        if not self.alternating_opt['enable']:
            return True  # Normal optimization
        
        # Update global epoch counter
        self.alternating_opt['global_epoch_counter'] = epoch
        
        # Check if we should enter joint phase
        if (self.alternating_opt['enable_joint_phases'] and 
            epoch % self.alternating_opt['joint_phase_frequency'] == 0 and
            self.alternating_opt['current_phase'] != 'joint'):
            self.alternating_opt['current_phase'] = 'joint'
            self.alternating_opt['phase_counter'] = 1
            logger.info(f"🤝 Epoch {epoch + 1}: Entering JOINT optimization phase")
            return 'joint'  # Special flag for joint optimization
            
        phase = self.alternating_opt['current_phase']
        counter = self.alternating_opt['phase_counter']
        
        if phase == 'joint':
            if counter >= self.alternating_opt['joint_phase_duration']:
                # Exit joint phase, return to main
                self.alternating_opt['current_phase'] = 'main'
                self.alternating_opt['phase_counter'] = 1
                logger.info(f"📊 Epoch {epoch + 1}: Exiting joint phase, returning to MAIN optimization")
                return True
            else:
                self.alternating_opt['phase_counter'] += 1
                logger.info(f"🤝 Epoch {epoch + 1}: JOINT optimization ({counter+1}/{self.alternating_opt['joint_phase_duration']})")
                return 'joint'
        elif phase == 'main':
            if counter >= self.alternating_opt['main_epochs']:
                # Switch to filter phase
                self.alternating_opt['current_phase'] = 'filter'
                self.alternating_opt['phase_counter'] = 1
                logger.info(f"🎛️ Epoch {epoch + 1}: Switching to FILTER optimization phase")
                return False
            else:
                self.alternating_opt['phase_counter'] += 1
                logger.info(f"📊 Epoch {epoch + 1}: MAIN network optimization ({counter+1}/{self.alternating_opt['main_epochs']})")
                return True
        else:  # filter phase
            if counter >= self.alternating_opt['filter_epochs']:
                # Switch to main phase
                self.alternating_opt['current_phase'] = 'main'
                self.alternating_opt['phase_counter'] = 1
                logger.info(f"📊 Epoch {epoch + 1}: Switching to MAIN network optimization phase")
                return True
            else:
                self.alternating_opt['phase_counter'] += 1
                logger.info(f"🎛️ Epoch {epoch + 1}: FILTER optimization ({counter+1}/{self.alternating_opt['filter_epochs']})")
                return False
    
    def _freeze_main_network(self):
        """Freeze main network parameters for filter-only optimization"""
        for name, param in self.model.named_parameters():
            if not ('breakpoint' in name or 'transition_width' in name or 'filter_bank' in name):
                param.requires_grad = False
    
    def _unfreeze_main_network(self):
        """Unfreeze main network parameters"""
        for name, param in self.model.named_parameters():
            param.requires_grad = True
    
    def _test_filter_candidates(self):
        """Test multiple filter candidates using validation set"""
        if not self.alternating_opt['validation_feedback']:
            return
            
        current_breakpoint = self.model.combined_log_linear_spec.breakpoint.item()
        current_transition_width = self.model.combined_log_linear_spec.transition_width.item()
        
        # Generate candidates around current value
        search_range_bp = current_breakpoint * self.alternating_opt['search_range_factor']
        search_range_tw = current_transition_width * self.alternating_opt['search_range_factor']
        
        candidates_bp = [
            current_breakpoint,
            max(20.0, current_breakpoint - search_range_bp * 0.5),
            min(8000.0, current_breakpoint + search_range_bp * 0.5),
            max(20.0, current_breakpoint - search_range_bp * 1.5),
            min(8000.0, current_breakpoint + search_range_bp * 1.5)
        ]
        
        candidates_tw = [
            current_transition_width,
            max(10.0, current_transition_width - search_range_tw * 0.5),
            min(500.0, current_transition_width + search_range_tw * 0.5),
            max(10.0, current_transition_width - search_range_tw * 1.5),
            min(500.0, current_transition_width + search_range_tw * 1.5)
        ]
        
        # Test each candidate combination
        best_candidate_bp = current_breakpoint
        best_candidate_tw = current_transition_width
        best_score = float('inf')
        
        original_breakpoint = current_breakpoint
        original_transition_width = current_transition_width
        self.model.eval()
        
        # Total combinations to test
        total_tests = len(candidates_bp) * len(candidates_tw)
        test_count = 0
        
        for bp in candidates_bp:
            for tw in candidates_tw:
                test_count += 1
                
                # Set candidate parameters
                self.model.combined_log_linear_spec.breakpoint.data = torch.tensor(bp, device=self.device)
                self.model.combined_log_linear_spec.transition_width.data = torch.tensor(tw, device=self.device)
                
                # Forza il ricalcolo della filter bank nel modello
                if hasattr(self.model, 'update_filters'):
                    self.model.update_filters()
                
                # Test on small validation subset (fast evaluation)
                val_loss = 0.0
                val_samples = 0
                max_val_batches = self.alternating_opt.get('validation_batches_per_candidate', 10)
                
                with torch.no_grad():
                    for batch_idx, (audio, hard_labels, soft_labels) in enumerate(self.val_loader):
                        if batch_idx >= max_val_batches:
                            break
                            
                        audio, hard_labels, soft_labels = audio.to(self.device), hard_labels.to(self.device), soft_labels.to(self.device)
                        
                        logits = self.model(audio)
                        
                        if hasattr(self, 'loss_type') and self.loss_type == 'focal':
                            loss = self.criterion(logits, hard_labels)
                        else:
                            loss, _, _ = self.criterion(logits, hard_labels, soft_labels)
                        
                        val_loss += loss.item() * audio.size(0)
                        val_samples += audio.size(0)
                
                avg_loss = val_loss / val_samples if val_samples > 0 else float('inf')
                
                if avg_loss < best_score:
                    best_score = avg_loss
                    best_candidate_bp = bp
                    best_candidate_tw = tw
                    
                logger.info(f"    🧪 Candidate {test_count}/{total_tests}: BP={bp:.1f}Hz, TW={tw:.1f} → Loss: {avg_loss:.4f}")
        
        # Set best candidate
        self.model.combined_log_linear_spec.breakpoint.data = torch.tensor(best_candidate_bp, device=self.device)
        self.model.combined_log_linear_spec.transition_width.data = torch.tensor(best_candidate_tw, device=self.device)
        self.model.train()
        
        logger.info(f"🏆 Best filter candidate: BP={best_candidate_bp:.1f}Hz, TW={best_candidate_tw:.1f}Hz (was BP={original_breakpoint:.1f}Hz, TW={original_transition_width:.1f}Hz)")
        
        # Track history
        self.alternating_opt['filter_history'].append({
            'epoch': self.epoch,
            'breakpoint': best_candidate_bp,
            'transition_width': best_candidate_tw,
            'score': best_score,
            'candidates_tested': total_tests
        })

    def _normalize_class_name(self, class_name):
        """Normalize class names for soft label matching"""
        # Try both space and underscore versions
        normalized_names = [
            class_name,                          # Original
            class_name.replace('_', ' '),        # Underscore to space  
            class_name.replace(' ', '_'),        # Space to underscore
            class_name.replace('_', ' ').title(), # Title case with spaces
            class_name.replace(' ', '_').title()  # Title case with underscores
        ]
        return normalized_names
    
    def _find_soft_label_with_normalization(self, soft_labels_dict, class_name):
        """Find soft label trying different name normalizations"""
        normalized_names = self._normalize_class_name(class_name)
        
        for name in normalized_names:
            if name in soft_labels_dict:
                return soft_labels_dict[name]
        
        # If not found, return None (will trigger uniform distribution)
        return None
    
    def _compute_adaptive_distillation_loss(self, logits, hard_labels, soft_labels):
        """Compute distillation loss with adaptive handling of missing soft labels"""
        
        # Detect samples with uniform soft labels (missing teacher predictions)
        batch_size, num_classes = soft_labels.shape
        uniform_value = 1.0 / num_classes
        tolerance = 0.01
        
        # Find samples with uniform distribution (missing soft labels)
        uniform_mask = torch.all(
            torch.abs(soft_labels - uniform_value) < tolerance, 
            dim=1
        )
        
        # Adaptive alpha: reduce for samples with uniform soft labels
        base_alpha = self.config.distillation.get('alpha', 0.4)
        adaptive_alpha = torch.full((batch_size,), base_alpha, device=self.device)
        adaptive_alpha[uniform_mask] *= 0.1  # Drastically reduce alpha for uniform samples
        
        # Log statistics
        num_uniform = uniform_mask.sum().item()
        if num_uniform > 0:
            logger.debug(f"Found {num_uniform}/{batch_size} samples with uniform soft labels, reducing alpha")
        
        # Compute losses with adaptive alpha
        loss, hard_loss, soft_loss = self.criterion(logits, hard_labels, soft_labels)
        
        # Apply adaptive weighting
        # Standard: loss = (1-alpha) * hard_loss + alpha * soft_loss
        # Adaptive: use different alpha per sample
        
        hard_weight = 1 - adaptive_alpha.mean()
        soft_weight = adaptive_alpha.mean()
        
        adaptive_loss = hard_weight * hard_loss + soft_weight * soft_loss
        
        return adaptive_loss, hard_loss, soft_loss
    
    def _patch_soft_label_loading(self):
        """Patch the dataset to improve soft label matching with name normalization"""
        # Check if we have a class mapping in config
        class_mapping = self.config.dataset.get('soft_label_class_mapping', {})
        
        if class_mapping:
            logger.info("🔧 Applying soft label class name mapping from config...")
            logger.info(f"Class mapping: {class_mapping}")
            
            # Apply mapping to dataset if accessible
            if hasattr(self, 'train_loader') and hasattr(self.train_loader.dataset, 'class_to_idx'):
                dataset = self.train_loader.dataset
                
                # Update class_to_idx mapping
                if hasattr(dataset, 'class_to_idx'):
                    original_mapping = dataset.class_to_idx.copy()
                    logger.info(f"Original class_to_idx: {original_mapping}")
                    
                    # Apply soft label mapping
                    for config_name, soft_label_name in class_mapping.items():
                        if config_name in original_mapping:
                            # Keep the original index but note the mapping
                            logger.debug(f"Mapping: '{config_name}' (config) → '{soft_label_name}' (soft labels)")
                    
                    logger.info("✅ Class mapping applied successfully")
                    return True
        else:
            logger.info("🔧 Applying automatic soft label name normalization...")
            # Fallback to automatic normalization
            if hasattr(self, 'train_loader') and hasattr(self.train_loader.dataset, 'soft_labels'):
                dataset = self.train_loader.dataset
                
                # Try to access soft labels dict if available
                if hasattr(dataset, 'soft_labels') and isinstance(dataset.soft_labels, dict):
                    original_soft_labels = dataset.soft_labels.copy()
                    
                    # Create normalized lookup table
                    normalized_lookup = {}
                    for file_path, soft_label in original_soft_labels.items():
                        normalized_lookup[file_path] = soft_label
                    
                    # Add normalized class names to metadata if accessible
                    if hasattr(dataset, 'class_names') or hasattr(dataset, 'classes'):
                        class_names = getattr(dataset, 'class_names', getattr(dataset, 'classes', []))
                        
                        logger.info(f"Original class names: {class_names}")
                        
                        # Apply normalization to class mappings
                        normalized_classes = []
                        for class_name in class_names:
                            normalized_names = self._normalize_class_name(class_name)
                            normalized_classes.extend(normalized_names)
                            logger.debug(f"Class '{class_name}' → {normalized_names}")
                        
                        logger.info(f"✅ Soft label normalization applied for {len(class_names)} classes")
                        return True
                        
        logger.warning("Cannot patch soft label loading - dataset structure not accessible or no mapping available")
        return False
    
    def setup_data(self):
        """Setup data loaders"""
        # Get soft labels path from config
        soft_labels_path = self.config.dataset.get('soft_labels_path', 'soft_labels_complete')
        
        # Create train loader
        self.train_loader, train_dataset = create_distillation_dataloader(
            config=self.config.dataset,
            soft_labels_path=soft_labels_path,
            split='train'
        )
        
        # Create validation loader
        self.val_loader, val_dataset = create_distillation_dataloader(
            config=self.config.dataset,
            soft_labels_path=soft_labels_path,
            split='val'
        )
        
        # Create test loader
        self.test_loader, test_dataset = create_distillation_dataloader(
            config=self.config.dataset,
            soft_labels_path=soft_labels_path,
            split='test'
        )
        
        # Set num_classes based on dataset
        if hasattr(train_dataset, 'num_classes'):
            self.num_classes = train_dataset.num_classes
        elif hasattr(train_dataset, 'classes'):
            self.num_classes = len(train_dataset.classes)
        else:
            # Fallback: use config
            allowed_classes = self.config.dataset.get('allowed_bird_classes', [])
            use_no_bird = self.config.dataset.get('use_no_bird_class', True)
            self.num_classes = len(allowed_classes) + (1 if use_no_bird else 0)
        
        # === CLASS NAMES DEBUGGING ===
        logger.info("=== CLASS NAMES DEBUGGING ===")
        
        # Get class names directly from config
        self.class_names = list(self.config.dataset.get('allowed_bird_classes', []))
        if self.config.dataset.get('use_no_bird_class', False):
            self.class_names.append('No Bird')
        
        logger.info(f"✅ Class names from config: {self.class_names}")
        logger.info(f"✅ Number of classes: {self.num_classes}")
        logger.info(f"✅ Class names length: {len(self.class_names)}")
        logger.info("=== END CLASS NAMES DEBUGGING ===")
        
        logger.info(f"✅ Detected {self.num_classes} classes for training")
        
        return train_dataset, val_dataset, test_dataset
        
    def _original_setup_data(self):
        """Original data setup method"""
        # Get soft labels path from config
        soft_labels_path = self.config.dataset.get('soft_labels_path', 'soft_labels_complete')
        
        # Create train loader - pass dataset config section
        self.train_loader, train_dataset = create_distillation_dataloader(
            config=self.config.dataset,  # Pass dataset section instead of full config
            soft_labels_path=soft_labels_path,
            split='train'
        )
        
        # Create validation loader
        self.val_loader, val_dataset = create_distillation_dataloader(
            config=self.config.dataset,  # Pass dataset section instead of full config
            soft_labels_path=soft_labels_path,
            split='val'
        )
        
        # Create test loader
        self.test_loader, test_dataset = create_distillation_dataloader(
            config=self.config.dataset,  # Pass dataset section instead of full config
            soft_labels_path=soft_labels_path,
            split='test'
        )
        
        # CRUCIALE: Imposta num_classes basandosi sul dataset
        if hasattr(train_dataset, 'num_classes'):
            self.num_classes = train_dataset.num_classes
        elif hasattr(train_dataset, 'classes'):
            self.num_classes = len(train_dataset.classes)
        else:
            # Fallback: usa config o calcola dal dataset
            allowed_classes = self.config.dataset.get('allowed_bird_classes', [])
            use_no_bird = self.config.dataset.get('use_no_bird_class', True)
            self.num_classes = len(allowed_classes) + (1 if use_no_bird else 0)
        
        logger.info(f"✅ Detected {self.num_classes} classes for training")
        
        return train_dataset, val_dataset, test_dataset

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
    
        logger.info(f"--- Distillation Training Completed ---")
        logger.info(f"Final test accuracy: {test_acc:.4f}")
        logger.info(f"Results and plots saved to: {output_dir}")
    except Exception as e:
        logger.exception(f"An error occurred during training or testing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 