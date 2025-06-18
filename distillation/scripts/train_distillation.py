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
        
        pbar = tqdm(self.val_loader, desc=f"Epoch {self.epoch} [Val]", unit="batch")
        with torch.no_grad():
            for audio, hard_labels, soft_labels in pbar:
                # Move to device
                audio = audio.to(self.device)
                hard_labels = hard_labels.to(self.device)
                soft_labels = soft_labels.to(self.device)
                
                # Forward pass
                logits = self.model(audio)
                
                # Compute distillation loss (only hard loss matters for val accuracy)
                loss, _, _ = self.criterion(logits, hard_labels, soft_labels)
                
                # Statistics
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = logits.max(1)
                total += hard_labels.size(0)
                correct += predicted.eq(hard_labels).sum().item()
        
                pbar.set_postfix(
                    loss=loss.item(), 
                    acc=f"{(100. * correct / total):.2f}%"
                )
        
        # Calculate average loss and accuracy (convert to percentage like train_acc)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total  # Convert to percentage for consistency
        
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
        """Test the model on the test set"""
        logger.info("Starting testing...")
        
        # Load best model for testing
        model_path = self.output_dir / f"{self.config.experiment_name}_best_model.pth"
        if model_path.exists():
            logger.info(f"Loading best model from {model_path} for testing...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No best model found. Testing with the model from the last epoch.")
        
        self.model.eval()
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Testing", unit="batch")
            for audio, hard_labels, _ in pbar:
                audio = audio.to(self.device)
                hard_labels = hard_labels.to(self.device)
                
                logits = self.model(audio)
                _, predicted = logits.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(hard_labels.cpu().numpy())
        
        # Calculate metrics
        self.test_acc = sum(1 for i, j in zip(all_true, all_preds) if i == j) / len(all_true)
        logger.info(f"Test Accuracy: {self.test_acc:.4f}")
        
        # Generate classification report
        report_str = classification_report(all_true, all_preds, target_names=self.class_names, zero_division=0)
        report_dict = classification_report(all_true, all_preds, target_names=self.class_names, zero_division=0, output_dict=True)
        logger.info(f"Test Report:\n{report_str}")
        
        # Save confusion matrix
        cm_png_path = self.output_dir / "confusion_matrix.png"
        cm_csv_path = self.output_dir / "confusion_matrix.csv"
        save_confusion_matrix(all_true, all_preds, self.class_names, cm_png_path, cm_csv_path)
        
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