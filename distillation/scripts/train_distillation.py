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
from omegaconf import DictConfig, OmegaConf
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

    # Save PNG
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    try:
        plt.savefig(png_path)
        logger.info(f"Confusion matrix PNG saved to {png_path}")
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
        
        # Create train loader
        self.train_loader, train_dataset = create_distillation_dataloader(
            self.config.dataset, self.soft_labels_path, split='train'
        )
        
        # Create validation loader
        self.val_loader, val_dataset = create_distillation_dataloader(
            self.config.dataset, self.soft_labels_path, split='val'
        )
        
        # Create test loader
        self.test_loader, test_dataset = create_distillation_dataloader(
            self.config.dataset, self.soft_labels_path, split='test'
        )
        
        # Log dataset info
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")
        
        # Get soft labels info and extract number of classes
        soft_info = train_dataset.get_soft_labels_info()
        self.num_classes = soft_info['num_classes']
        self.class_names = train_dataset.get_classes()
        
        logger.info(f"Soft labels info: {soft_info}")
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class names: {self.class_names}")
        
        return train_dataset, val_dataset, test_dataset
    
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
        """Setup optimizer and scheduler"""
        logger.info("Setting up optimizer and scheduler...")
        
        # Optimizer
        self.optimizer = hydra.utils.instantiate(
            self.config.optimizer,
            params=self.model.parameters()
        )
        
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
            max_samples = loss_config.get('weight_calculation_samples', 1000)  # Default: use only 1000 samples
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
                    _, hard_label, _ = train_dataset[i]
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
            
            if successful_samples == 0:
                logger.error("No samples could be loaded for class weight computation!")
                # Return equal weights as fallback
                num_classes = self.num_classes
                class_weights = [1.0] * num_classes
            else:
                # Scale counts to represent full dataset if we used sampling
                if use_sampling and samples_to_scan < dataset_size:
                    scaling_factor = dataset_size / samples_to_scan
                    for label in class_counts:
                        class_counts[label] = int(class_counts[label] * scaling_factor)
                    logger.info(f"Scaled sample counts by {scaling_factor:.2f} to estimate full dataset distribution")
                
                # Convert to ordered list
                max_class = max(class_counts.keys()) if class_counts else 0
                class_counts_list = [class_counts.get(i, 0) for i in range(max_class + 1)]
                
                logger.info(f"Estimated class distribution: {class_counts_list}")
                
                # Compute class weights using inverse frequency
                total_samples = sum(class_counts_list)
                num_classes = len(class_counts_list)
                
                # Avoid division by zero with minimum count threshold
                min_count_threshold = max(1, total_samples // (num_classes * 100))  # At least 1% of average
                class_weights = []
                for count in class_counts_list:
                    effective_count = max(count, min_count_threshold)  # Avoid zero division
                    weight = total_samples / (num_classes * effective_count)
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
                    'class_distribution': class_counts_list if 'class_counts_list' in locals() else [],
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
            
            # Track filter parameters if using combined_log_linear
            if (self.config.model.get('spectrogram_type') == 'combined_log_linear' and 
                hasattr(self.model, 'combined_log_linear_spec') and 
                self.model.combined_log_linear_spec is not None):
                current_breakpoint = self.model.combined_log_linear_spec.breakpoint.item()
                current_transition_width = self.model.combined_log_linear_spec.transition_width.item()
                self.breakpoint_history.append(current_breakpoint)
                self.transition_width_history.append(current_transition_width)
                logger.info(f"Epoch {self.epoch} - Filter params: Breakpoint={current_breakpoint:.2f}Hz, Transition Width={current_transition_width:.2f}")
            
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