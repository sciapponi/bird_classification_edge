from datasets import BirdSoundDataset, ESC50Dataset, create_combined_dataset, download_and_extract_bird_dataset, download_and_extract_esc50
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import check_model, check_forward_pass, count_precise_macs
import hydra 
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import logging
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

@hydra.main(version_base=None, config_path='./config', config_name='bird_classification')
def train(cfg: DictConfig):
    """
    Main training function for the bird classification model.
    
    Args:
        cfg: Hydra configuration
    """
    # Set up logging
    log = logging.getLogger(__name__)
    experiment_name = cfg.experiment_name
    log.info(f"Experiment: {experiment_name}")
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Set random seeds for reproducibility
    seed = cfg.training.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")
    
    # Download datasets if needed
    if cfg.dataset.get("download_datasets", False):
        log.info("Checking for datasets...")
        if not os.path.exists(cfg.dataset.bird_data_dir):
            log.info("Bird dataset not found. Please download and organize bird sound data manually.")
        
        # Download ESC-50 dataset
        esc50_dir = download_and_extract_esc50()
        cfg.dataset.esc50_dir = esc50_dir
    
    # Create combined datasets
    log.info("Creating datasets...")
    train_dataset = create_combined_dataset(
        bird_data_dir=cfg.dataset.bird_data_dir,
        esc50_dir=cfg.dataset.esc50_dir,
        allowed_bird_classes=cfg.dataset.allowed_bird_classes,
        bird_to_background_ratio=cfg.dataset.bird_to_background_ratio,
        use_augmentation=cfg.dataset.augmentation.enabled,
        target_sr=cfg.dataset.sample_rate,
        clip_duration=cfg.dataset.clip_duration,
        subset="training"
    )
    
    val_dataset = create_combined_dataset(
        bird_data_dir=cfg.dataset.bird_data_dir,
        esc50_dir=cfg.dataset.esc50_dir,
        allowed_bird_classes=cfg.dataset.allowed_bird_classes,
        bird_to_background_ratio=cfg.dataset.bird_to_background_ratio,
        use_augmentation=False,  # No augmentation for validation
        target_sr=cfg.dataset.sample_rate,
        clip_duration=cfg.dataset.clip_duration,
        subset="validation"
    )
    
    test_dataset = create_combined_dataset(
        bird_data_dir=cfg.dataset.bird_data_dir,
        esc50_dir=cfg.dataset.esc50_dir,
        allowed_bird_classes=cfg.dataset.allowed_bird_classes,
        bird_to_background_ratio=cfg.dataset.bird_to_background_ratio,
        use_augmentation=False,  # No augmentation for testing
        target_sr=cfg.dataset.sample_rate,
        clip_duration=cfg.dataset.clip_duration,
        subset="testing"
    )
    
    log.info(f"Training samples: {len(train_dataset)}")
    log.info(f"Validation samples: {len(val_dataset)}")
    log.info(f"Testing samples: {len(test_dataset)}")
    
    # Create DataLoaders
    batch_size = cfg.training.get('batch_size', 64)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    num_classes = len(cfg.dataset.allowed_bird_classes) + 1  # +1 for the non-bird class
    model = instantiate(cfg.model, num_classes=num_classes).to(device)
    
    # Check model architecture and complexity
    log.info(f"Model architecture:")
    log.info(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,}")
    log.info(f"Trainable parameters: {trainable_params:,}")
    
    # Compute MACs (Multi-Accumulate operations)
    log.info("Computing model complexity...")
    with torch.no_grad():
        # Get a sample input
        inputs, _ = next(iter(train_loader))
        sample_input = inputs[:1].to(device)  # Just use one sample
        
        try:
            macs = count_precise_macs(model, sample_input)
            log.info(f"MACs per inference: {macs:,}")
        except Exception as e:
            log.warning(f"Could not compute MACs: {e}")
    
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, model.parameters())
    
    # Learning rate scheduler
    scheduler = instantiate(cfg.scheduler, optimizer)
    
    # Training loop with early stopping
    num_epochs = cfg.training.epochs
    best_val_acc = 0.0
    best_model_path = f"{output_dir}/best_{experiment_name}.pth"
    
    # Early stopping parameters
    patience = cfg.training.get('patience', 10)
    min_delta = cfg.training.get('min_delta', 0.001)
    early_stop_counter = 0
    best_val_loss = float('inf')
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    log.info("Starting training...")
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.0 * train_correct / train_total:.2f}%"
            })
        
        train_loss = train_loss / train_total
        train_acc = 100.0 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Log metrics
        log.info(f"Epoch {epoch+1}/{num_epochs}")
        log.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        log.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        log.info(f"Learning Rate: {current_lr:.6f}")
        
        # Check for improvement
        if val_loss + min_delta < best_val_loss:
            log.info(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            log.info(f"No improvement in validation loss for {early_stop_counter}/{patience} epochs")
            
            if early_stop_counter >= patience:
                log.info(f"Early stopping triggered after {epoch+1} epochs!")
                break
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            log.info(f"New best model saved with val acc: {val_acc:.2f}%")
    
    # Plot training history
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_history.png", dpi=300)
        log.info(f"Training history plot saved to {output_dir}/training_history.png")
    except Exception as e:
        log.warning(f"Could not create training history plot: {e}")
    
    # Evaluate on test set
    log.info("\nEvaluating on test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
            
            # Save for confusion matrix
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    
    test_loss = test_loss / test_total
    test_acc = 100.0 * test_correct / test_total
    
    log.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Compute and save confusion matrix
    try:
        # Get class names (combine bird classes and "non-bird")
        class_names = cfg.dataset.allowed_bird_classes + ["non-bird"]
        
        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        # Save confusion matrix as CSV
        cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.2f}%')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
        log.info(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    except Exception as e:
        log.warning(f"Could not create confusion matrix: {e}")
    
    # Save model summary and configuration
    with open(f"{output_dir}/model_summary.txt", "w") as f:
        f.write(f"Model: {cfg.model._target_}\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        try:
            f.write(f"MACs per inference: {macs:,}\n")
        except:
            pass
        f.write(f"\nBest validation accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Test accuracy: {test_acc:.2f}%\n")
    
    # Save final results
    results = {
        "experiment_name": experiment_name,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }
    
    import json
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    log.info(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    log.info(f"Test accuracy: {test_acc:.2f}%")
    log.info(f"Results saved to {output_dir}")
    
    return results

if __name__ == "__main__":
    train()