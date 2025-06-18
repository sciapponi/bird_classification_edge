#!/usr/bin/env python3
"""
Student Model Prediction Script

This script generates predictions using the trained student model
on audio files directly without additional preprocessing.
"""

import torch
import torch.nn.functional as F
import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import hydra

# Add project root to path BEFORE other project imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import Improved_Phi_GRU_ATT

# Keep only essential, non-conflicting imports at the global level
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


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
            # Debug environment variables
            cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            logger.info(f"Environment CUDA_VISIBLE_DEVICES: '{cuda_devices}'")
            # This check now happens before librosa is ever imported
            is_available = torch.cuda.is_available()
            logger.info(f"PyTorch CUDA available: {is_available}")
            if is_available:
                logger.info(f"PyTorch CUDA device count: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    logger.info(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
            
            self.device = torch.device("cuda" if is_available else "cpu")
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
            
            # Load state dict - ALWAYS load on CPU first to avoid CUDA device mismatch
            checkpoint = torch.load(self.model_path, map_location='cpu')
            
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
    
    def _training_aligned_preprocessing(self, audio_path: str) -> torch.Tensor:
        """
        Audio preprocessing aligned with training pipeline using extract_calls.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        # --- LOCAL IMPORT ---
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from datasets.audio_utils import extract_call_segments
        import torch
        import numpy as np

        try:
            # Use same preprocessing as training: extract_calls=True
            call_intervals, segments, _, _, _ = extract_call_segments(
                audio_path, 
                clip_duration=3.0,
                sr=32000, 
                lowcut=150.0,   # Same as training
                highcut=15500.0, # Slightly below Nyquist to avoid filter edge case
                verbose=False
            )
            
            if segments and len(segments) > 0:
                # Use first extracted segment (same as training)
                audio_data = segments[0]
                audio_tensor = torch.from_numpy(audio_data).float()
                
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
                    
                # Ensure correct length (same as training)
                target_length = int(32000 * 3.0)
                if audio_tensor.shape[1] > target_length:
                    audio_tensor = audio_tensor[:, :target_length]
                elif audio_tensor.shape[1] < target_length:
                    padding = target_length - audio_tensor.shape[1]
                    audio_tensor = torch.nn.functional.pad(audio_tensor, (0, padding))
                    
            else:
                # Fallback: if no calls detected, use random clip (same as training fallback)
                logger.warning(f"No calls detected in {audio_path}, using fallback preprocessing")
                import librosa
                y, sr = librosa.load(audio_path, sr=32000, duration=3.0)
                
                # Apply same bandpass filter as training
                from scipy import signal
                nyquist = sr / 2
                low = 150.0 / nyquist
                high = 15500.0 / nyquist
                b, a = signal.butter(4, [low, high], btype='band')
                y = signal.filtfilt(b, a, y)
                
                # Normalize
                if np.max(np.abs(y)) > 0:
                    y = y / np.max(np.abs(y))
                
                # Ensure length
                target_length = int(32000 * 3.0)
                if len(y) < target_length:
                    y = np.pad(y, (0, target_length - len(y)), mode='constant')
                elif len(y) > target_length:
                    y = y[:target_length]
                
                audio_tensor = torch.FloatTensor(y).unsqueeze(0)
            
            return audio_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to preprocess audio {audio_path}: {e}")
            # Return silence as fallback
            target_length = int(32000 * 3.0)
            return torch.zeros(1, target_length).to(self.device)
    
    def _load_audio_like_training(self, audio_path: str) -> torch.Tensor:
        """
        Exact copy of load_audio method from bird_dataset.py
        This does NOT apply bandpass filtering - just like training!
        """
        import torchaudio
        import torchaudio.transforms as T
        
        try:
            waveform, sr_orig = torchaudio.load(audio_path)
        except Exception as e:
            logger.error(f"ERROR loading audio file {audio_path}: {e}. Returning zeros.")
            target_len_samples = int(32000 * 3.0)
            return torch.zeros((1, target_len_samples))

        # Resample if necessary (same as training)
        if sr_orig != 32000:
            resampler = T.Resample(sr_orig, 32000)
            waveform = resampler(waveform)
            
        # Convert to mono if necessary (same as training)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Ensure correct length (same as training)
        waveform = self._ensure_length(waveform, target_length=int(32000 * 3.0))
        
        return waveform
    
    def _ensure_length(self, waveform: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Exact copy of ensure_length method from bird_dataset.py
        """
        import torch.nn.functional as F
        
        # Standardize waveform shape to [channels, time]
        if waveform.ndim == 3:
            if waveform.shape[0] == 1:
                waveform = waveform.squeeze(0)
            elif waveform.shape[1] == 1:
                waveform = waveform.squeeze(1)
            else:
                if waveform.shape[0] > 1 and waveform.ndim == 3:
                    waveform = waveform[0] 
                    if waveform.ndim == 1:
                        waveform = waveform.unsqueeze(0)
                else:
                    return torch.zeros(1, target_length)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        if waveform.ndim != 2:
            return torch.zeros(1, target_length)

        current_length = waveform.shape[1]
        
        if current_length == target_length:
            return waveform
        elif current_length > target_length:
            return waveform[:, :target_length]
        else:
            padding_needed = target_length - current_length
            return F.pad(waveform, (0, padding_needed))
    
    def predict_single(self, audio_path: Union[str, Path]) -> Dict:
        """
        Generate prediction for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio using training-aligned pipeline
            audio_tensor = self._training_aligned_preprocessing(str(audio_path))
            
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
                          ground_truth_df: "pd.DataFrame",
                          audio_base_path: str = "") -> "pd.DataFrame":
    """
    Generate predictions for audio files listed in a ground truth DataFrame.
    
    Args:
        predictor: Student model predictor instance
        ground_truth_df: DataFrame with 'audio_path' column
        audio_base_path: Base path to prepend to audio paths
        
    Returns:
        DataFrame with predictions added
    """
    # --- LOCAL IMPORT ---
    import pandas as pd

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
    # --- LOCAL IMPORT ---
    import hydra
    import pandas as pd

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
        output_path = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir, "student_predictions.csv")
        
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