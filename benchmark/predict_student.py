#!/usr/bin/env python3
"""
Student Model Prediction Script

This script generates predictions using the trained student model
on audio files directly without additional preprocessing.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torch
import torch.nn.functional as F
import librosa
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import Improved_Phi_GRU_ATT

logger = logging.getLogger(__name__)


class StudentModelPredictor:
    """
    Generates predictions using the trained student model.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str,
                 device: str = "auto",
                 confidence_threshold: float = 0.1):
        """
        Initialize student model predictor.
        
        Args:
            model_path: Path to trained model file
            config_path: Path to model configuration file
            device: Device to run model on ("auto", "cpu", "cuda")
            confidence_threshold: Minimum confidence threshold
        """
        self.model_path = model_path
        self.config_path = config_path
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Load model
        self.model = self._load_model()
        
        # Extract class names
        self.class_names = self.config.dataset.allowed_bird_classes + ["no_birds"]
        
        logger.info(f"Student model initialized with {len(self.class_names)} classes")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def _load_config(self) -> DictConfig:
        """Load model configuration."""
        try:
            config = OmegaConf.load(self.config_path)
            logger.info(f"Loaded configuration from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model."""
        try:
            # Extract model parameters from config
            model_config = self.config.model
            
            # Initialize model with proper parameters
            model = Improved_Phi_GRU_ATT(
                num_classes=model_config.num_classes,
                spectrogram_type=model_config.spectrogram_type,
                sample_rate=self.config.dataset.sample_rate,
                n_mel_bins=model_config.n_mel_bins,
                n_linear_filters=model_config.n_linear_filters,
                f_min=self.config.dataset.lowcut,
                f_max=self.config.dataset.highcut,
                hidden_dim=model_config.hidden_dim,
                n_fft=model_config.n_fft,
                hop_length=model_config.hop_length,
                matchbox=model_config.matchbox,
                breakpoint=model_config.matchbox.breakpoint,
                transition_width=model_config.matchbox.transition_width
            )
            
            # Load state dict
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            
            # Set to evaluation mode and move to device
            model.eval()
            model.to(self.device)
            
            logger.info(f"Loaded model from: {self.model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _simple_audio_preprocessing(self, audio_path: str) -> torch.Tensor:
        """
        Simple audio preprocessing without problematic filters.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        try:
            # Load audio with librosa
            y, sr = librosa.load(audio_path, sr=32000, duration=3.0)
            
            # Ensure minimum length
            min_length = int(32000 * 3.0)  # 3 seconds at 32kHz
            if len(y) < min_length:
                # Pad with zeros
                y = np.pad(y, (0, min_length - len(y)), mode='constant')
            elif len(y) > min_length:
                # Truncate
                y = y[:min_length]
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(y).unsqueeze(0)  # Add batch dimension
            
            return audio_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_path}: {e}")
            raise
    
    def predict_single(self, audio_path: Union[str, Path]) -> Dict:
        """
        Generate prediction for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio
            audio_tensor = self._simple_audio_preprocessing(str(audio_path))
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model(audio_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_idx = predicted_idx.item()
                predicted_class = self.class_names[predicted_idx]
            
            return {
                'audio_path': str(audio_path),
                'predicted_class': predicted_class,
                'predicted_idx': predicted_idx,
                'confidence': confidence,
                'all_probabilities': probabilities.cpu().numpy().tolist()[0]
            }
            
        except Exception as e:
            logger.error(f"Failed to predict {audio_path}: {e}")
            return {
                'audio_path': str(audio_path),
                'predicted_class': "no_birds",
                'predicted_idx': len(self.class_names) - 1,
                'confidence': 0.0,
                'all_probabilities': [0.0] * len(self.class_names),
                'error': str(e)
            }
    
    def predict_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Generate predictions for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        logger.info(f"Starting student model predictions for {len(audio_paths)} files")
        
        for audio_path in tqdm(audio_paths, desc="Student predictions"):
            prediction = self.predict_single(audio_path)
            predictions.append(prediction)
        
        logger.info(f"Completed student model predictions")
        return predictions


def predict_from_dataframe(predictor: StudentModelPredictor, 
                          ground_truth_df: pd.DataFrame,
                          audio_base_path: str = "") -> pd.DataFrame:
    """
    Generate predictions for audio files listed in a ground truth DataFrame.
    
    Args:
        predictor: Student model predictor instance
        ground_truth_df: DataFrame with 'audio_path' column
        audio_base_path: Base path to prepend to audio paths
        
    Returns:
        DataFrame with predictions added
    """
    predictions = []
    
    # Progress bar for student model predictions
    for _, row in tqdm(ground_truth_df.iterrows(), 
                      total=len(ground_truth_df), 
                      desc="🤖 Student model predictions", 
                      unit="files"):
        audio_path = os.path.join(audio_base_path, row['audio_path']) if audio_base_path else row['audio_path']
        
        prediction = predictor.predict_single(audio_path)
        
        predictions.append({
            'audio_path': row['audio_path'],
            'ground_truth': row.get('label', row.get('true_label', 'unknown')),
            'student_prediction': prediction['predicted_class'],
            'student_confidence': prediction['confidence'],
            'student_predicted_idx': prediction['predicted_idx'],
            'student_all_probabilities': prediction['all_probabilities']
        })
    
    return pd.DataFrame(predictions)


@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Main function for student model prediction."""
    
    # Get original working directory (before Hydra changes it)
    from hydra.core.hydra_config import HydraConfig
    original_cwd = HydraConfig.get().runtime.cwd
    
    logger.info("Starting student model prediction")
    logger.info(f"Configuration: {cfg.student_model}")
    
    # Initialize predictor
    predictor = StudentModelPredictor(
        model_path=os.path.join(original_cwd, cfg.benchmark.paths.student_model),
        config_path=os.path.join(original_cwd, cfg.benchmark.paths.student_config),
        device=cfg.student_model.inference.device,
        confidence_threshold=cfg.student_model.inference.confidence_threshold
    )
    
    try:
        # Load ground truth
        ground_truth_path = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir, "ground_truth.csv")
        
        if not os.path.exists(ground_truth_path):
            logger.error(f"Ground truth file not found: {ground_truth_path}")
            return
        
        ground_truth_df = pd.read_csv(ground_truth_path)
        logger.info(f"Loaded ground truth with {len(ground_truth_df)} samples")
        
        # Generate predictions
        predictions_df = predict_from_dataframe(
            predictor, 
            ground_truth_df, 
            audio_base_path=original_cwd
        )
        
        # Save predictions
        benchmark_dir = os.path.join(original_cwd, "benchmark") 
        output_path = os.path.join(benchmark_dir, cfg.benchmark.paths.predictions_dir, "student_predictions.csv")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"Student predictions saved to: {output_path}")
        logger.info(f"Prediction summary:")
        logger.info(f"  Total predictions: {len(predictions_df)}")
        
        # Log prediction distribution
        pred_counts = predictions_df['student_prediction'].value_counts()
        for species, count in pred_counts.items():
            logger.info(f"  {species}: {count}")
        
        # Log average confidence
        avg_confidence = predictions_df['student_confidence'].mean()
        logger.info(f"  Average confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Error during student model prediction: {e}")
        raise


if __name__ == "__main__":
    main() 