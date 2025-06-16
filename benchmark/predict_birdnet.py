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

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

logger = logging.getLogger(__name__)


class BirdNETPredictor:
    """
    Generates predictions using BirdNET-Analyzer with species filtering.
    """
    
    def __init__(self, 
                 target_species: List[str],
                 confidence_threshold: float = 0.1,
                 species_list_path: Optional[str] = None):
        """
        Initialize BirdNET predictor.
        
        Args:
            target_species: List of target species (underscore format)
            confidence_threshold: Minimum confidence for predictions
            species_list_path: Path to custom species list file
        """
        self.target_species = target_species
        self.confidence_threshold = confidence_threshold
        
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
            self.analyzer = Analyzer(custom_species_list_path=self.species_list_path)
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
    
    def _create_species_list(self, species_labels):
        """Create temporary species list file for BirdNET filtering"""
        species_mapping = {
            'Bubo_bubo': 'Bubo bubo_Eurasian Eagle-Owl',
            'Apus_apus': 'Apus apus_Common Swift', 
            'Certhia_familiaris': 'Certhia familiaris_Eurasian Treecreeper',
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
            
            # Determine top prediction
            if species_confidences:
                top_species = max(species_confidences.keys(), key=lambda x: species_confidences[x])
                top_confidence = species_confidences[top_species]
            else:
                # No detections - classify as no_birds
                top_species = "no_birds"
                top_confidence = 0.0
            
            return {
                'audio_path': str(audio_path),
                'predicted_class': top_species,
                'confidence': top_confidence,
                'all_detections': all_detections,
                'species_confidences': species_confidences,
                'num_detections': len(all_detections)
            }
            
        except Exception as e:
            logger.error(f"Failed to predict {audio_path}: {e}")
            return {
                'audio_path': str(audio_path),
                'predicted_class': "no_birds",
                'confidence': 0.0,
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
                          audio_base_path: str = "") -> pd.DataFrame:
    """
    Generate predictions for audio files listed in a ground truth DataFrame.
    
    Args:
        predictor: BirdNET predictor instance
        ground_truth_df: DataFrame with 'audio_path' column
        audio_base_path: Base path to prepend to audio paths
        
    Returns:
        DataFrame with predictions added
    """
    predictions = []
    
    # Progress bar for BirdNET predictions
    for _, row in tqdm(ground_truth_df.iterrows(), 
                      total=len(ground_truth_df), 
                      desc="🦅 BirdNET predictions", 
                      unit="files"):
        audio_path = os.path.join(audio_base_path, row['audio_path']) if audio_base_path else row['audio_path']
        
        prediction = predictor.predict_single(audio_path)
        
        predictions.append({
            'audio_path': row['audio_path'],
            'ground_truth': row.get('label', row.get('true_label', 'unknown')),
            'birdnet_prediction': prediction['predicted_class'],
            'birdnet_confidence': prediction['confidence'],
            'birdnet_detections': prediction['num_detections'],
            'birdnet_all_species': prediction['species_confidences']
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
    
    # Initialize predictor
    predictor = BirdNETPredictor(
        target_species=cfg.student_model.classes.allowed_species,
        confidence_threshold=cfg.birdnet.confidence_threshold
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