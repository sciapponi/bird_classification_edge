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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from distillation.losses.distillation_loss import DistillationLoss, AdaptiveDistillationLoss
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

def save_confusion_matrix(cm, class_names, filepath):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()

class DistillationTrainer:
    def __init__(self, config, soft_labels_path):
        self.config = config
        self.soft_labels_path = soft_labels_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.hard_losses = []
        self.soft_losses = []
        
        logger.info(f"Initialized trainer on device: {self.device}")
    
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
        
        # Log soft labels info
        soft_info = train_dataset.get_soft_labels_info()
        logger.info(f"Soft labels info: {soft_info}")
        
        return train_dataset, val_dataset, test_dataset
    
    def setup_model(self, num_classes):
        """Setup student model"""
        logger.info("Setting up student model...")
        
        # Create model with config parameters
        model_config = self.config.model
        
        # Convert matchbox config to regular dict to avoid Hydra struct issues
        matchbox_config = dict(model_config.get('matchbox', {}))
        
        self.model = Improved_Phi_GRU_ATT(
            num_classes=num_classes,
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
        """Setup distillation loss function"""
        logger.info("Setting up distillation loss...")
        
        # Get distillation parameters from config
        alpha = self.config.distillation.get('alpha', 0.5)
        temperature = self.config.distillation.get('temperature', 4.0)
        adaptive = self.config.distillation.get('adaptive', False)
        
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
        
        logger.info(f"Using {'Adaptive' if adaptive else 'Standard'} DistillationLoss")
        logger.info(f"Alpha: {alpha}, Temperature: {temperature}")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch} [Train]", unit="batch")
        for audio, hard_labels, soft_labels in pbar:
            # Move to device
            audio = audio.to(self.device)
            hard_labels = hard_labels.to(self.device)
            soft_labels = soft_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(audio)
            
            # Compute distillation loss
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
            pbar.set_postfix(
                loss=loss.item(), 
                hard_loss=hard_loss.item(), 
                soft_loss=soft_loss.item(),
                acc=f"{(100. * correct / total):.2f}%"
            )
        
        avg_loss = total_loss / len(self.train_loader)
        avg_hard_loss = total_hard_loss / len(self.train_loader)
        avg_soft_loss = total_soft_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_hard_loss, avg_soft_loss, accuracy
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]", unit="batch")
        with torch.no_grad():
            for audio, hard_labels, soft_labels in pbar:
                # Move to device
                audio = audio.to(self.device)
                hard_labels = hard_labels.to(self.device)
                soft_labels = soft_labels.to(self.device)
                
                # Forward pass
                logits = self.model(audio)
                
                # Compute distillation loss
                loss, hard_loss, soft_loss = self.criterion(logits, hard_labels, soft_labels)
                
                # Statistics
                total_loss += loss.item()
                total_hard_loss += hard_loss.item()
                total_soft_loss += soft_loss.item()
                
                # Accuracy
                _, predicted = logits.max(1)
                total += hard_labels.size(0)
                correct += predicted.eq(hard_labels).sum().item()

                # Set postfix for tqdm progress bar
                pbar.set_postfix(
                    loss=loss.item(),
                    acc=f"{(100. * correct / total):.2f}%"
                )
        
        avg_loss = total_loss / len(self.val_loader)
        avg_hard_loss = total_hard_loss / len(self.val_loader)
        avg_soft_loss = total_soft_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_hard_loss, avg_soft_loss, accuracy
    
    def train(self):
        """Main training loop"""
        logger.info("Starting distillation training...")
        
        # Setup early stopping
        early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            min_delta=self.config.training.min_delta
        )
        
        for epoch in range(self.config.training.epochs):
            self.epoch = epoch
            
            # Train epoch
            train_loss, train_hard_loss, train_soft_loss, train_acc = self.train_epoch()
            
            # Validate epoch
            val_loss, val_hard_loss, val_soft_loss, val_acc = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                if hasattr(self.scheduler, 'step'):
                    if 'ReduceLROnPlateau' in str(type(self.scheduler)):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            # Adaptive alpha adjustment
            if hasattr(self.criterion, 'adapt_alpha'):
                self.criterion.adapt_alpha(val_acc)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.hard_losses.append(train_hard_loss)
            self.soft_losses.append(train_soft_loss)
            
            # Log epoch results
            logger.info(
                f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%'
            )
            logger.info(
                f'  Hard Loss: {train_hard_loss:.4f}, Soft Loss: {train_soft_loss:.4f}, '
                f'Alpha: {self.criterion.alpha:.3f}'
            )
            
            # Save checkpoint if best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_loss, val_acc,
                    filepath='best_distillation_model.pt'
                )
                logger.info(f"New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Early stopping check
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def test(self):
        """Test the best model"""
        logger.info("Testing best model...")
        
        # Load best model
        checkpoint = load_checkpoint('best_distillation_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        pbar = tqdm(self.test_loader, desc="Testing", unit="batch")
        with torch.no_grad():
            for audio, hard_labels, soft_labels in pbar:
                audio = audio.to(self.device)
                hard_labels = hard_labels.to(self.device)
                
                logits = self.model(audio)
                _, predicted = logits.max(1)
                
                total += hard_labels.size(0)
                correct += predicted.eq(hard_labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(hard_labels.cpu().numpy())

                pbar.set_postfix(acc=f"{(100. * correct / total):.2f}%")
        
        test_acc = 100. * correct / total
        logger.info(f"Test Accuracy: {test_acc:.2f}%")
        
        # Generate detailed metrics
        class_names = self.test_loader.dataset.get_classes()
        if len(set(all_labels)) > len(class_names):
             # Fallback if the number of unique labels in the test set is greater
             # than the number of classes provided by the dataset.
            class_names = [f"Class_{i}" for i in range(len(set(all_labels)))]

        report = classification_report(
            all_labels, all_predictions, target_names=class_names, 
            zero_division=0, labels=range(len(class_names))
        )
        logger.info(f"Classification Report:\n{report}")
        
        # Save confusion matrix
        cm = confusion_matrix(all_labels, all_predictions, labels=range(len(class_names)))
        save_confusion_matrix(cm, class_names, 'distillation_confusion_matrix.png')
        
        return test_acc, report, cm
    
    def save_training_plots(self):
        """Save training history plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plots
        axes[0, 0].plot(self.train_losses, label='Train Total')
        axes[0, 0].plot(self.val_losses, label='Val Total')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plots
        axes[0, 1].plot(self.train_accs, label='Train')
        axes[0, 1].plot(self.val_accs, label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Hard vs Soft loss
        axes[1, 0].plot(self.hard_losses, label='Hard Loss')
        axes[1, 0].plot(self.soft_losses, label='Soft Loss')
        axes[1, 0].set_title('Hard vs Soft Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Alpha evolution (if adaptive)
        if hasattr(self.criterion, 'alpha'):
            alpha_values = [self.criterion.alpha] * len(self.train_losses)  # Simplified
            axes[1, 1].plot(alpha_values)
            axes[1, 1].set_title('Alpha Evolution')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Alpha')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('distillation_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Training plots saved to distillation_training_history.png")

@hydra.main(version_base=None, config_path="../configs", config_name="distillation_config")
def main(cfg: DictConfig):
    """Main training function"""
    logger.info("Starting knowledge distillation training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Check for soft labels path
    soft_labels_path = cfg.get('soft_labels_path', 'soft_labels')
    if not Path(soft_labels_path).exists():
        logger.error(f"Soft labels directory not found: {soft_labels_path}")
        logger.error("Please run extract_soft_labels.py first!")
        return
    
    # Create trainer
    trainer = DistillationTrainer(cfg, soft_labels_path)
    
    # Setup
    train_dataset, val_dataset, test_dataset = trainer.setup_data()
    
    # Get num_classes from the dataset, which now reads it from soft labels metadata
    soft_labels_info = train_dataset.get_soft_labels_info()
    num_classes = soft_labels_info.get('num_classes')
    
    if not num_classes:
        logger.error("Could not determine the number of classes from soft labels.")
        return

    trainer.setup_model(num_classes)
    trainer.setup_optimizer()
    trainer.setup_criterion()
    
    # Train
    trainer.train()
    
    # Test
    test_acc, report, cm = trainer.test()
    
    # Save plots
    trainer.save_training_plots()
    
    logger.info("Distillation training completed successfully!")
    logger.info(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main() 