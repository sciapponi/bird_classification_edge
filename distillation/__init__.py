"""
Knowledge Distillation package for Bird Sound Classification.

This package contains components for knowledge distillation using BirdNET as teacher model:
- scripts: Main execution scripts
- losses: Distillation loss functions
- datasets: Dataset classes with soft labels
- configs: Configuration files
"""

__version__ = "1.0.0"
__author__ = "Bird Classification Edge Team"

# Main components
from .losses.distillation_loss import DistillationLoss, AdaptiveDistillationLoss
from .datasets.distillation_dataset import DistillationBirdSoundDataset, create_distillation_dataloader

__all__ = [
    "DistillationLoss",
    "AdaptiveDistillationLoss", 
    "DistillationBirdSoundDataset",
    "create_distillation_dataloader"
] 