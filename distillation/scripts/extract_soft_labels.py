#!/usr/bin/env python3
"""
Extract soft labels from BirdNET for knowledge distillation.
This script processes all audio files in the dataset and generates soft labels
that will be used to train the student model.
"""

import os
import sys
import json
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

class SoftLabelExtractor:
    def __init__(self, species_list_path="distillation/species.txt", confidence_threshold=0.05):
        """
        Initialize the soft label extractor.
        
        Args:
            species_list_path: Path to species.txt file with target species
            confidence_threshold: Minimum confidence for including predictions
        """
        self.analyzer = Analyzer(custom_species_list_path=species_list_path)
        self.confidence_threshold = confidence_threshold
        
        # Load target species and create mapping
        self.target_species = self.load_target_species(species_list_path)
        self.species_to_idx = {species: idx for idx, species in enumerate(self.target_species)}
        self.num_classes = len(self.target_species)
        
        print(f"Loaded {self.num_classes} target species")
        print(f"Using confidence threshold: {confidence_threshold}")
    
    
    # carica i nomi delle specie target dalla cartella del dataset
    def load_target_species(self, species_list_path):
        """Load scientific names from species.txt and match with actual dataset"""
        import os
        
        # Get species from actual dataset directories
        dataset_path = "bird_sound_dataset_processed"
        actual_species = []
        print(f"Dataset_path: {dataset_path}")
        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                if os.path.isdir(os.path.join(dataset_path, item)) and not item.startswith('.'):
                    # Convert directory name to scientific name (e.g., "Bubo_bubo" -> "Bubo bubo")
                    scientific_name = item.replace('_', ' ')
                    actual_species.append(scientific_name)
        # Also add a "non-bird" class at the end
        actual_species.append("non-bird")
        print(f"Found {len(actual_species)} species in dataset: {actual_species}")
        
        
        
        return actual_species
    
    def extract_soft_labels(self, audio_path):
        """
        Extract soft labels for a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            numpy array of shape (num_classes,) with soft label probabilities
        """
        try:
            # Analyze audio with BirdNET
            recording = Recording(
                self.analyzer,
                audio_path,
                min_conf=self.confidence_threshold
            )
            recording.analyze()
            
            # Initialize soft label vector
            soft_labels = np.zeros(self.num_classes, dtype=np.float32)
            
            # Process detections
            total_confidence = 0.0
            found_target_species = False
            
            for detection in recording.detections:
                scientific_name = detection['scientific_name']
                confidence = detection['confidence']
                
                if scientific_name in self.species_to_idx:
                    idx = self.species_to_idx[scientific_name]
                    # Use max confidence if multiple detections for same species
                    soft_labels[idx] = max(soft_labels[idx], confidence)
                    total_confidence += confidence
                    found_target_species = True
                else:
                    # This is a species not in our target list
                    # Add confidence to non-bird class (last index)
                    if "non-bird" in self.species_to_idx:
                        non_bird_idx = self.species_to_idx["non-bird"]
                        soft_labels[non_bird_idx] = max(soft_labels[non_bird_idx], confidence * 0.1)  # Reduced weight
            
            # If no target species found but BirdNET detected something,
            # treat as non-bird or background noise
            if not found_target_species and len(recording.detections) > 0:
                if "non-bird" in self.species_to_idx:
                    non_bird_idx = self.species_to_idx["non-bird"]
                    # Give moderate confidence to non-bird class
                    soft_labels[non_bird_idx] = max(soft_labels[non_bird_idx], 0.3)
            
            # If nothing detected at all, slight preference for non-bird
            if len(recording.detections) == 0:
                if "non-bird" in self.species_to_idx:
                    non_bird_idx = self.species_to_idx["non-bird"]
                    soft_labels[non_bird_idx] = 0.1
            
            return soft_labels
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Return slight non-bird preference on error
            soft_labels = np.zeros(self.num_classes, dtype=np.float32)
            if "non-bird" in self.species_to_idx:
                non_bird_idx = self.species_to_idx["non-bird"]
                soft_labels[non_bird_idx] = 0.1
            return soft_labels
    
    def process_dataset(self, dataset_path, output_path, max_files_per_class=None):
        """
        Process entire dataset and save soft labels.
        
        Args:
            dataset_path: Path to bird_sound_dataset directory
            output_path: Path to save soft labels
            max_files_per_class: Limit files per species (for testing)
        """
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        soft_labels_data = {}
        failed_files = []
        
        # Get all audio files
        audio_files = []
        for species_dir in dataset_path.iterdir():
            if species_dir.is_dir():
                # Get all files in the directory (no extension filtering)
                species_files = [f for f in species_dir.iterdir() if f.is_file()]
                
                # Limit files per class if specified
                if max_files_per_class:
                    species_files = species_files[:max_files_per_class]
                
                audio_files.extend(species_files)
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Process each audio file
        for audio_path in tqdm(audio_files, desc="Extracting soft labels"):
            try:
                # Extract soft labels
                soft_labels = self.extract_soft_labels(audio_path)
                
                # Store with relative path as key
                rel_path = str(audio_path.relative_to(dataset_path))
                soft_labels_data[rel_path] = soft_labels.tolist()
                
            except Exception as e:
                print(f"Failed to process {audio_path}: {e}")
                failed_files.append(str(audio_path))
        
        # Save results
        output_file = output_path / "soft_labels.json"
        with open(output_file, 'w') as f:
            json.dump(soft_labels_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'num_classes': self.num_classes,
            'target_species': self.target_species,
            'species_to_idx': self.species_to_idx,
            'confidence_threshold': self.confidence_threshold,
            'total_files_processed': len(soft_labels_data),
            'failed_files': failed_files
        }
        
        metadata_file = output_path / "soft_labels_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Soft labels saved to {output_file}")
        print(f"Processed {len(soft_labels_data)} files successfully")
        print(f"Failed: {len(failed_files)} files")
        
        return soft_labels_data, metadata

def main():
    parser = argparse.ArgumentParser(description="Extract soft labels from BirdNET")
    parser.add_argument("--dataset_path", type=str, default="bird_sound_dataset_processed",
                       help="Path to bird sound dataset")
    parser.add_argument("--output_path", type=str, default="soft_labels",
                       help="Output directory for soft labels")
    parser.add_argument("--species_list", type=str, default="distillation/species.txt",
                       help="Path to species list file")
    parser.add_argument("--confidence_threshold", type=float, default=0.05,
                       help="Minimum confidence threshold")
    parser.add_argument("--max_files_per_class", type=int, default=None,
                       help="Limit files per class (for testing)")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = SoftLabelExtractor(
        species_list_path=args.species_list,
        confidence_threshold=args.confidence_threshold
    )
    
    # Process dataset
    soft_labels_data, metadata = extractor.process_dataset(
        args.dataset_path,
        args.output_path,
        args.max_files_per_class
    )
    
    print("Soft label extraction completed!")

if __name__ == "__main__":
    main() 