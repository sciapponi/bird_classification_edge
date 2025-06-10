# Knowledge Distillation Package

This package implements knowledge distillation for bird sound classification using BirdNET as the teacher model and a lightweight student model for edge deployment.

## 📁 Package Structure

```
distillation/
├── README.md                    # This file
├── __init__.py                  # Package initialization
├── scripts/                     # Executable scripts
│   ├── __init__.py
│   ├── extract_soft_labels.py  # Extract soft labels from BirdNET
│   └── train_distillation.py   # Train with knowledge distillation
├── datasets/                    # Dataset classes with soft labels
│   ├── __init__.py
│   └── distillation_dataset.py # Dataset that loads soft labels
├── losses/                      # Distillation loss functions
│   ├── __init__.py
│   └── distillation_loss.py    # Distillation and adaptive losses
└── configs/                     # Configuration files
    ├── distillation_config.yaml # Main distillation config
    └── test_distillation.yaml   # Test configuration
```

## 🔄 Workflow

### Step 1: Extract Soft Labels from BirdNET

```bash
# From project root
python extract_soft_labels.py --confidence_threshold 0.05

# Or directly
python distillation/scripts/extract_soft_labels.py --confidence_threshold 0.05
```

This creates:
- `soft_labels/soft_labels.json` - Soft label vectors for each audio file
- `soft_labels/soft_labels_metadata.json` - Metadata about classes and extraction

### Step 2: Train Student Model with Distillation

```bash
# From project root
python train_distillation.py

# Or directly  
python distillation/scripts/train_distillation.py

# With custom config
python train_distillation.py --config-name=test_distillation
```

## 🎯 Key Components

### DistillationLoss
Combines hard labels (ground truth) and soft labels (from teacher):
- `alpha`: Weight balance between hard and soft loss
- `temperature`: Temperature scaling for smoother distributions
- `AdaptiveDistillationLoss`: Automatically adjusts alpha based on validation performance

### DistillationBirdSoundDataset
Extends the base BirdSoundDataset to also load soft labels from BirdNET for each audio sample.

### Configuration
- **distillation_config.yaml**: Full distillation training configuration
- **test_distillation.yaml**: Quick test with subset of data

## 📊 Expected Benefits

- **Improved Accuracy**: Student learns from teacher's rich knowledge
- **Better Generalization**: Soft labels provide information about similar species
- **Edge Deployment**: Student model remains lightweight (~53k parameters)
- **Challenging Classes**: Helps with difficult species like Poecile montanus

## 🔧 Hyperparameter Tuning

Key parameters to experiment with:
- `distillation.alpha` (0.1-0.7): Higher = more teacher influence
- `distillation.temperature` (1.0-8.0): Higher = softer distributions  
- `training.lr` (0.0001-0.001): Often needs lower LR for distillation
- `confidence_threshold` (0.03-0.1): Minimum confidence for soft labels

## 📈 Monitoring Training

The training script logs:
- Total loss (combined hard + soft)
- Hard loss (standard classification)
- Soft loss (KL divergence with teacher)
- Alpha value (if adaptive)
- Validation accuracy

## 🎛️ Usage Examples

```python
# Load distillation components
from distillation import DistillationLoss, DistillationBirdSoundDataset

# Create loss function
loss_fn = DistillationLoss(alpha=0.3, temperature=4.0)

# Create dataset with soft labels
dataset = DistillationBirdSoundDataset(
    soft_labels_path="soft_labels",
    root_dir="bird_sound_dataset",
    subset="training"
)

# Training step
student_logits = model(audio)
total_loss, hard_loss, soft_loss = loss_fn(
    student_logits, hard_labels, teacher_soft_labels
)
``` 