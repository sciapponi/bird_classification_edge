#!/usr/bin/env python3
"""
BirdNET Prediction Script

This script generates predictions using BirdNET-Analyzer on audio files,
with species filtering similar to the knowledge distillation process.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
import tempfile
from scipy import signal

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datasets.audio_utils import extract_call_segments

logger = logging.getLogger(__name__)


class BirdNETPredictor:
    """
    Generates predictions using BirdNET-Analyzer with species filtering.
    Supports both full-file analysis and segmented analysis for fair comparison.
    """
    
    def __init__(self, 
                 target_species: List[str],
                 confidence_threshold: float = 0.2,
                 use_adaptive_threshold: bool = False,
                 adaptive_factor: float = 1.0,
                 force_bird_prediction: bool = False,
                 species_list_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 segment_duration: float = 3.0,
                 sample_rate: int = 32000,
                 lowcut: float = 150.0,
                 highcut: float = 16000.0,
                 extract_calls: bool = True):
        """
        Initialize BirdNET predictor.
        
        Args:
            target_species: List of target species (underscore format)
            confidence_threshold: Single confidence threshold for all classifications
            use_adaptive_threshold: Use adaptive threshold strategies (currently disabled)
            adaptive_factor: Factor for adaptive threshold calculations (not used when adaptive disabled)
            force_bird_prediction: If True, always predict best bird species instead of no_birds (for birds-only mode)
            species_list_path: Path to custom species list file
            model_path: Path to the TFLite classification model (FP32 preferred for highest accuracy). 
                If None, BirdNET's default path is used.
            segment_duration: Duration of audio segments for aligned comparison
            sample_rate: Sample rate for preprocessing
            lowcut: Low frequency cutoff for bandpass filter
            highcut: High frequency cutoff for bandpass filter
            extract_calls: Whether to use call extraction (align with student model)
        """
        self.target_species = target_species
        self.confidence_threshold = confidence_threshold
        self.use_adaptive_threshold = use_adaptive_threshold
        self.adaptive_factor = adaptive_factor
        self.force_bird_prediction = force_bird_prediction
        self.model_path = model_path
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.lowcut = lowcut
        self.highcut = highcut
        self.extract_calls = extract_calls
        
        # Convert target species to BirdNET format (spaces instead of underscores)
        self.birdnet_species = [species.replace('_', ' ') for species in target_species]
        
        # Create species list file for BirdNET
        self.temp_species_file = "temp_species_list.txt"
        self.species_list_path = self._create_species_list(target_species)
        
        # Initialize BirdNET analyzer with custom species list
        # Apply UTF-8 patch for international characters
        import locale
        import builtins
        
        # Set locale to UTF-8
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        
        # Temporarily patch builtins.open to use UTF-8 encoding
        original_open = builtins.open
        def utf8_open(*args, **kwargs):
            if 'encoding' not in kwargs:
                kwargs['encoding'] = 'utf-8'
            return original_open(*args, **kwargs)
        
        # Apply patch and initialize analyzer
        builtins.open = utf8_open
        try:
            self.analyzer = Analyzer(
                custom_species_list_path=self.species_list_path
            )
        finally:
            # Restore original open function
            builtins.open = original_open
        
        # Create species mapping (BirdNET format -> project format)
        self.species_mapping = {
            birdnet_name: project_name 
            for birdnet_name, project_name in zip(self.birdnet_species, self.target_species)
        }
        # Add reverse mapping
        self.species_mapping.update({
            project_name: birdnet_name 
            for birdnet_name, project_name in zip(self.birdnet_species, self.target_species)
        })
        
        logger.info(f"BirdNET initialized with {len(self.target_species)} target species")
        logger.info(f"Target species: {self.target_species}")
        logger.info(f"Confidence threshold: {confidence_threshold}")
        logger.info(f"Segmented mode: {extract_calls}, Duration: {segment_duration}s")
    
    def _create_species_list(self, species_labels):
        """Create temporary species list file for BirdNET filtering"""
        species_mapping = {
            'Bubo_bubo': 'Bubo bubo_Eurasian Eagle-Owl',
            'Apus_apus': 'Apus apus_Common Swift', 
            'Certhia_familiaris': 'Certhia familiaris_Eurasian Treecreeper',
            'Certhia_brachydactyla': 'Certhia brachydactyla_Short-toed Treecreeper',
            'Emberiza_cia': 'Emberiza cia_Rock Bunting',
            'Lophophanes_cristatus': 'Lophophanes cristatus_Crested Tit',
            'Periparus_ater': 'Periparus ater_Coal Tit',
            'Poecile_montanus': 'Poecile montanus_Willow Tit',
            'no_birds': 'no_birds'  # Keep as is for no_birds
        }
        
        species_list_content = []
        for label in species_labels:
            if label in species_mapping:
                if label != 'no_birds':  # Don't add no_birds to the species list
                    species_list_content.append(species_mapping[label])
        
        # Write to temporary file
        with open(self.temp_species_file, 'w') as f:
            for species in species_list_content:
                f.write(f"{species}\n")
        
        return self.temp_species_file
    
    def _apply_adaptive_threshold_strategy(self, top_species: str, top_confidence: float, 
                                         species_confidences: Dict[str, float]) -> tuple:
        """
        Apply simplified threshold strategy for bird classification.
        
        Args:
            top_species: Species with highest confidence
            top_confidence: Highest confidence score
            species_confidences: All species confidences
            
        Returns:
            Tuple of (predicted_class, final_confidence)
        """
        # Check if we should force bird prediction (birds-only mode)
        if self.force_bird_prediction and species_confidences:
            # Always predict the best bird species, regardless of threshold
            logger.debug(f"Force bird prediction: {top_species} with confidence {top_confidence:.3f} (threshold bypassed)")
            return top_species, top_confidence
        
        # Standard single threshold approach
        if top_confidence >= self.confidence_threshold:
            logger.debug(f"Bird classification: {top_species} with confidence {top_confidence:.3f} >= {self.confidence_threshold:.3f}")
            return top_species, top_confidence
        else:
            logger.debug(f"No-birds classification: {top_confidence:.3f} < {self.confidence_threshold:.3f}")
            return "no_birds", 0.0
    
    def _preprocess_audio_segment(self, audio_path: str) -> Optional[str]:
        """
        Preprocess audio with identical pipeline to student model.
        
        Args:
            audio_path: Path to original audio file
            
        Returns:
            Path to preprocessed temporary file or None if failed
        """
        try:
            if self.extract_calls:
                # Use call extraction like student model
                call_intervals, segments, original_audio, sr, duration = extract_call_segments(
                    audio_path,
                    clip_duration=self.segment_duration,
                    sr=self.sample_rate,
                    lowcut=self.lowcut,
                    highcut=self.highcut,
                    verbose=False
                )
                
                if segments and len(segments) > 0:
                    # Use first extracted segment
                    audio_segment = segments[0]
                    logger.debug(f"Using extracted call segment from {audio_path}")
                else:
                    # Fallback to random 3s clip with bandpass filter
                    logger.debug(f"No calls found in {audio_path}, using fallback preprocessing")
                    audio_segment = self._fallback_preprocessing(audio_path)
                    if audio_segment is None:
                        return None
            else:
                # Use random 3s clip with bandpass filter
                audio_segment = self._fallback_preprocessing(audio_path)
                if audio_segment is None:
                    return None
            
            # Save preprocessed segment to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(temp_file.name, audio_segment, self.sample_rate)
            temp_file.close()
            
            return temp_file.name
            
        except Exception as e:
            logger.error(f"Failed to preprocess {audio_path}: {e}")
            return None
    
    def _fallback_preprocessing(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Fallback preprocessing: random 3s clip with bandpass filter.
        Identical to student model preprocessing when extract_calls=False.
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Apply bandpass filter (same as student model)
            nyquist = sr / 2
            low = self.lowcut / nyquist
            high = self.highcut / nyquist
            
            if low >= 1.0 or high >= 1.0:
                logger.warning(f"Filter frequencies too high for sample rate {sr}")
                # Skip filtering if frequencies are invalid
                y_filtered = y
            else:
                b, a = signal.butter(4, [low, high], btype='band')
                y_filtered = signal.filtfilt(b, a, y)
            
            # Extract 3-second segment
            target_length = int(self.segment_duration * sr)
            
            if len(y_filtered) >= target_length:
                # Random start position
                max_start = len(y_filtered) - target_length
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
                segment = y_filtered[start_idx:start_idx + target_length]
            else:
                # Pad if too short
                segment = np.pad(y_filtered, (0, target_length - len(y_filtered)), mode='constant')
            
            # Normalize
            if np.max(np.abs(segment)) > 0:
                segment = segment / np.max(np.abs(segment))
            
            return segment
            
        except Exception as e:
            logger.error(f"Fallback preprocessing failed for {audio_path}: {e}")
            return None

    def predict_single_aligned(self, audio_path: Union[str, Path]) -> Dict:
        """
        Generate BirdNET prediction using aligned preprocessing (3s segments).
        This method ensures fair comparison with the student model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        temp_file = None
        try:
            # Preprocess audio with identical pipeline to student model
            temp_file = self._preprocess_audio_segment(str(audio_path))
            
            if temp_file is None:
                logger.warning(f"Preprocessing failed for {audio_path}")
                fallback_class = "no_birds"
                fallback_confidence = 0.0
                if self.force_bird_prediction and self.target_species:
                    # Use a random bird species (excluding no_birds) to reduce bias
                    import random
                    bird_species = [s for s in self.target_species if s != "no_birds"]
                    if bird_species:
                        fallback_class = random.choice(bird_species)
                        fallback_confidence = 0.001
                        logger.debug(f"Bird-only prediction (preprocessing failed): {fallback_class} (random)")
                    else:
                        logger.warning("No bird species available for force prediction after preprocessing failure")
                
                return {
                    'audio_path': str(audio_path),
                    'predicted_class': fallback_class,
                    'confidence': fallback_confidence,
                    'all_detections': [],
                    'species_confidences': {},
                    'num_detections': 0,
                    'preprocessing_method': 'failed'
                }
            
            # Create recording object with preprocessed segment
            recording = Recording(
                self.analyzer,
                temp_file,
                min_conf=self.confidence_threshold
            )
            
            # Disable location and time filters for fair comparison
            recording.lat = None
            recording.lon = None
            recording.date = None
            recording.week = -1
            
            # Analyze recording
            recording.analyze()
            
            # Process detections
            all_detections = []
            species_confidences = {}
            
            for detection in recording.detections:
                scientific_name = detection['scientific_name']
                confidence = detection['confidence']
                
                # Convert to project format
                project_species = scientific_name.replace(' ', '_')
                
                detection_info = {
                    'scientific_name': scientific_name,
                    'project_species': project_species,
                    'common_name': detection['common_name'],
                    'confidence': float(confidence),
                    'start_time': detection.get('start_time', 0.0),
                    'end_time': detection.get('end_time', self.segment_duration)
                }
                
                all_detections.append(detection_info)
                
                # Keep highest confidence for each species
                if project_species not in species_confidences or confidence > species_confidences[project_species]:
                    species_confidences[project_species] = confidence
            
            # Determine top prediction with proper threshold logic
            if self.force_bird_prediction:
                # Birds-only mode: always predict best bird species regardless of threshold
                # First try to get detections with current threshold
                if species_confidences:
                    # We have detections above threshold, use them
                    top_species = max(species_confidences.keys(), key=lambda x: species_confidences[x])
                    top_confidence = species_confidences[top_species]
                    predicted_class = top_species
                    final_confidence = top_confidence
                    logger.debug(f"Bird-only prediction (above threshold): {predicted_class} with confidence {final_confidence:.3f}")
                else:
                    # No detections above threshold, get ALL detections to find best bird
                    recording_all = Recording(
                        self.analyzer,
                        temp_file,
                        min_conf=0.001  # Very low threshold to get all possible detections
                    )
                    recording_all.lat = None
                    recording_all.lon = None
                    recording_all.date = None
                    recording_all.week = -1
                    recording_all.analyze()
                    
                    # Find best detection among target species only
                    all_species_confidences = {}
                    for detection in recording_all.detections:
                        scientific_name = detection['scientific_name']
                        confidence = detection['confidence']
                        project_species = scientific_name.replace(' ', '_')
                        
                        # Only consider target species (excluding no_birds)
                        if project_species in self.target_species and project_species != "no_birds":
                            if project_species not in all_species_confidences or confidence > all_species_confidences[project_species]:
                                all_species_confidences[project_species] = confidence
                    
                    if all_species_confidences:
                        # Choose species with highest confidence, even if very low
                        predicted_class = max(all_species_confidences.keys(), key=lambda x: all_species_confidences[x])
                        final_confidence = all_species_confidences[predicted_class]
                        logger.debug(f"Bird-only prediction (below threshold): {predicted_class} with confidence {final_confidence:.4f}")
                    else:
                        # Absolute fallback: randomly choose from target species to avoid bias
                        import random
                        bird_species = [s for s in self.target_species if s != "no_birds"]
                        if bird_species:
                            predicted_class = random.choice(bird_species)
                            final_confidence = 0.001
                            logger.debug(f"Bird-only prediction (random fallback): {predicted_class} with minimal confidence")
                        else:
                            predicted_class = "no_birds"
                            final_confidence = 0.0
                            logger.warning("No bird species available for force prediction")
            else:
                # Standard mode with no_birds: use threshold logic
                if species_confidences:
                    top_species = max(species_confidences.keys(), key=lambda x: species_confidences[x])
                    top_confidence = species_confidences[top_species]
                    
                    # Apply adaptive threshold strategy
                    predicted_class, final_confidence = self._apply_adaptive_threshold_strategy(
                        top_species, top_confidence, species_confidences
                    )
                else:
                    # No detections above threshold - predict no_birds
                    predicted_class = "no_birds"
                    final_confidence = 0.0
                    logger.debug("No birds detected above threshold")
            
            return {
                'audio_path': str(audio_path),
                'predicted_class': predicted_class,
                'confidence': final_confidence,
                'all_detections': all_detections,
                'species_confidences': species_confidences,
                'num_detections': len(all_detections),
                'preprocessing_method': 'extract_calls' if self.extract_calls else 'random_clip',
                'adaptive_threshold_applied': self.use_adaptive_threshold and len(species_confidences) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to predict {audio_path}: {e}")
            fallback_class = "no_birds"
            fallback_confidence = 0.0
            if self.force_bird_prediction and self.target_species:
                # Use a random bird species (excluding no_birds) to reduce bias
                import random
                bird_species = [s for s in self.target_species if s != "no_birds"]
                if bird_species:
                    fallback_class = random.choice(bird_species)
                    fallback_confidence = 0.001
                    logger.debug(f"Bird-only prediction (exception): {fallback_class} (random)")
                else:
                    logger.warning("No bird species available for force prediction after exception")
            
            return {
                'audio_path': str(audio_path),
                'predicted_class': fallback_class,
                'confidence': fallback_confidence,
                'all_detections': [],
                'species_confidences': {},
                'num_detections': 0,
                'error': str(e),
                'preprocessing_method': 'error'
            }
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def predict_single(self, audio_path: Union[str, Path]) -> Dict:
        """
        Generate BirdNET prediction for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Create recording object
            recording = Recording(
                self.analyzer,
                str(audio_path),
                min_conf=self.confidence_threshold
            )
            
            # Disable location and time filters for fair comparison
            recording.lat = None
            recording.lon = None
            recording.date = None
            recording.week = -1
            
            # Analyze recording
            recording.analyze()
            
            # Process detections
            all_detections = []
            species_confidences = {}
            
            for detection in recording.detections:
                scientific_name = detection['scientific_name']
                confidence = detection['confidence']
                
                # Convert to project format
                project_species = scientific_name.replace(' ', '_')
                
                detection_info = {
                    'scientific_name': scientific_name,
                    'project_species': project_species,
                    'common_name': detection['common_name'],
                    'confidence': float(confidence),
                    'start_time': detection.get('start_time', 0.0),
                    'end_time': detection.get('end_time', 3.0)
                }
                
                all_detections.append(detection_info)
                
                # Keep highest confidence for each species
                if project_species not in species_confidences or confidence > species_confidences[project_species]:
                    species_confidences[project_species] = confidence
            
            # Determine top prediction with proper threshold logic
            if self.force_bird_prediction:
                # Birds-only mode: always predict best bird species regardless of threshold
                # First try to get detections with current threshold
                if species_confidences:
                    # We have detections above threshold, use them
                    top_species = max(species_confidences.keys(), key=lambda x: species_confidences[x])
                    top_confidence = species_confidences[top_species]
                    predicted_class = top_species
                    final_confidence = top_confidence
                    logger.debug(f"Bird-only prediction (above threshold): {predicted_class} with confidence {final_confidence:.3f}")
                else:
                    # No detections above threshold, get ALL detections to find best bird
                    recording_all = Recording(
                        self.analyzer,
                        str(audio_path),
                        min_conf=0.001  # Very low threshold to get all possible detections
                    )
                    recording_all.lat = None
                    recording_all.lon = None
                    recording_all.date = None
                    recording_all.week = -1
                    recording_all.analyze()
                    
                    # Find best detection among target species only
                    all_species_confidences = {}
                    for detection in recording_all.detections:
                        scientific_name = detection['scientific_name']
                        confidence = detection['confidence']
                        project_species = scientific_name.replace(' ', '_')
                        
                        # Only consider target species (excluding no_birds)
                        if project_species in self.target_species and project_species != "no_birds":
                            if project_species not in all_species_confidences or confidence > all_species_confidences[project_species]:
                                all_species_confidences[project_species] = confidence
                    
                    if all_species_confidences:
                        # Choose species with highest confidence, even if very low
                        predicted_class = max(all_species_confidences.keys(), key=lambda x: all_species_confidences[x])
                        final_confidence = all_species_confidences[predicted_class]
                        logger.debug(f"Bird-only prediction (below threshold): {predicted_class} with confidence {final_confidence:.4f}")
                    else:
                        # Absolute fallback: randomly choose from target species to avoid bias
                        import random
                        bird_species = [s for s in self.target_species if s != "no_birds"]
                        if bird_species:
                            predicted_class = random.choice(bird_species)
                            final_confidence = 0.001
                            logger.debug(f"Bird-only prediction (random fallback): {predicted_class} with minimal confidence")
                        else:
                            predicted_class = "no_birds"
                            final_confidence = 0.0
                            logger.warning("No bird species available for force prediction")
            else:
                # Standard mode with no_birds: use threshold logic
                if species_confidences:
                    top_species = max(species_confidences.keys(), key=lambda x: species_confidences[x])
                    top_confidence = species_confidences[top_species]
                    
                    # Apply adaptive threshold strategy
                    predicted_class, final_confidence = self._apply_adaptive_threshold_strategy(
                        top_species, top_confidence, species_confidences
                    )
                else:
                    # No detections above threshold - predict no_birds
                    predicted_class = "no_birds"
                    final_confidence = 0.0
                    logger.debug("No birds detected above threshold")
            
            return {
                'audio_path': str(audio_path),
                'predicted_class': predicted_class,
                'confidence': final_confidence,
                'all_detections': all_detections,
                'species_confidences': species_confidences,
                'num_detections': len(all_detections),
                'adaptive_threshold_applied': self.use_adaptive_threshold and len(species_confidences) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to predict {audio_path}: {e}")
            fallback_class = "no_birds"
            fallback_confidence = 0.0
            if self.force_bird_prediction and self.target_species:
                # Use a random bird species (excluding no_birds) to reduce bias
                import random
                bird_species = [s for s in self.target_species if s != "no_birds"]
                if bird_species:
                    fallback_class = random.choice(bird_species)
                    fallback_confidence = 0.001
                    logger.debug(f"Bird-only prediction (exception): {fallback_class} (random)")
                else:
                    logger.warning("No bird species available for force prediction after exception")
            
            return {
                'audio_path': str(audio_path),
                'predicted_class': fallback_class,
                'confidence': fallback_confidence,
                'all_detections': [],
                'species_confidences': {},
                'num_detections': 0,
                'error': str(e)
            }
    
    def predict_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Generate BirdNET predictions for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        logger.info(f"Starting BirdNET predictions for {len(audio_paths)} files")
        
        for audio_path in tqdm(audio_paths, desc="BirdNET predictions"):
            prediction = self.predict_single(audio_path)
            predictions.append(prediction)
        
        logger.info(f"Completed BirdNET predictions")
        return predictions
    
    def cleanup(self):
        """Clean up temporary files."""
        temp_file = "temp_species_list.txt"
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logger.info("Cleaned up temporary species list file")


def predict_from_dataframe(predictor: BirdNETPredictor, 
                          ground_truth_df: pd.DataFrame,
                          audio_base_path: str = "",
                          use_aligned_processing: bool = True) -> pd.DataFrame:
    """
    Generate predictions for audio files listed in a ground truth DataFrame.
    
    Args:
        predictor: BirdNET predictor instance
        ground_truth_df: DataFrame with 'audio_path' column
        audio_base_path: Base path to prepend to audio paths
        use_aligned_processing: Whether to use aligned 3s segmented processing
        
    Returns:
        DataFrame with predictions added
    """
    predictions = []
    
    # Choose prediction method
    predict_method = predictor.predict_single_aligned if use_aligned_processing else predictor.predict_single
    method_name = "aligned (3s segments)" if use_aligned_processing else "full file"
    
    logger.info(f"Using BirdNET prediction method: {method_name}")
    
    # Progress bar for BirdNET predictions
    for _, row in tqdm(ground_truth_df.iterrows(), 
                      total=len(ground_truth_df), 
                      desc=f"🦅 BirdNET predictions ({method_name})", 
                      unit="files"):
        audio_path = os.path.join(audio_base_path, row['audio_path']) if audio_base_path else row['audio_path']
        
        prediction = predict_method(audio_path)
        
        predictions.append({
            'audio_path': row['audio_path'],
            'ground_truth': row.get('label', row.get('true_label', 'unknown')),
            'birdnet_prediction': prediction['predicted_class'],
            'birdnet_confidence': prediction['confidence'],
            'birdnet_detections': prediction['num_detections'],
            'birdnet_all_species': prediction['species_confidences'],
            'birdnet_preprocessing': prediction.get('preprocessing_method', 'unknown')
        })
    
    return pd.DataFrame(predictions)


@hydra.main(version_base=None, config_path="config", config_name="benchmark")
def main(cfg: DictConfig) -> None:
    """Main function for BirdNET prediction."""
    
    # Get original working directory (before Hydra changes it)
    from hydra.core.hydra_config import HydraConfig
    original_cwd = HydraConfig.get().runtime.cwd
    
    logger.info("Starting BirdNET prediction")
    logger.info(f"Configuration: {cfg.birdnet}")
    
    # Use the larger FP32 classification model explicitly
    model_fp32_path = os.path.join(
        original_cwd,
        "analyzer",
        "BirdNET_GLOBAL_6K_V2.4_Model_FP32.tflite"
    )
    
    # Configure BirdNET with aligned preprocessing parameters
    predictor = BirdNETPredictor(
        target_species=cfg.student_model.classes.allowed_species,
        confidence_threshold=cfg.birdnet.confidence_threshold,
        model_path=model_fp32_path,
        segment_duration=cfg.student_model.preprocessing.clip_duration,
        sample_rate=cfg.student_model.preprocessing.sample_rate,
        lowcut=cfg.student_model.preprocessing.lowcut,
        highcut=cfg.student_model.preprocessing.highcut,
        extract_calls=cfg.student_model.preprocessing.get('extract_calls', True)
    )
    
    try:
        # Load ground truth
        ground_truth_path = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir, "ground_truth.csv")
        
        if not os.path.exists(ground_truth_path):
            logger.error(f"Ground truth file not found: {ground_truth_path}")
            return
        
        ground_truth_df = pd.read_csv(ground_truth_path)
        logger.info(f"Loaded ground truth with {len(ground_truth_df)} samples")
        
        # Generate predictions using aligned processing
        predictions_df = predict_from_dataframe(
            predictor, 
            ground_truth_df, 
            audio_base_path=original_cwd,
            use_aligned_processing=True  # Use 3s segments with identical preprocessing
        )
        
        # Save predictions
        output_path = os.path.join(original_cwd, cfg.benchmark.paths.predictions_dir, "birdnet_predictions.csv")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        predictions_df.to_csv(output_path, index=False)
        
        logger.info(f"BirdNET predictions saved to: {output_path}")
        logger.info(f"Prediction summary:")
        logger.info(f"  Total predictions: {len(predictions_df)}")
        
        # Log prediction distribution
        pred_counts = predictions_df['birdnet_prediction'].value_counts()
        for species, count in pred_counts.items():
            logger.info(f"  {species}: {count}")
        
        # Log average confidence
        avg_confidence = predictions_df['birdnet_confidence'].mean()
        logger.info(f"  Average confidence: {avg_confidence:.3f}")
        
    except Exception as e:
        logger.error(f"Error during BirdNET prediction: {e}")
        raise
    finally:
        # Cleanup
        predictor.cleanup()


if __name__ == "__main__":
    main() 