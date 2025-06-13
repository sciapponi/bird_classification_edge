#!/usr/bin/env python3
"""
Structured Audio Segmentation Script for Benchmark Pipeline

Takes audio files organized in class subdirectories and splits them into 
fixed-duration segments while preserving class labels.

Structure:
  bird_sound_dataset/
  ├── Bubo_bubo/
  │   ├── file1.wav
  │   └── file2.wav
  ├── Certhia_familiaris/
  │   └── file3.wav
  └── ...
  
  augmented_dataset/no_birds/
  ├── empty_0000.wav
  └── empty_0001.wav
"""

import os
import argparse
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional, Dict
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StructuredAudioSegmenter:
    """
    Segments audio files organized in class subdirectories into fixed-duration clips.
    Automatically generates ground truth from folder structure.
    """
    
    def __init__(self, 
                 bird_audio_dir: str,
                 no_birds_dir: str,
                 output_dir: str, 
                 segment_duration: float = 3.0,
                 sample_rate: int = 32000,
                 overlap: float = 0.0,
                 min_duration: float = None,
                 max_files_per_class: int = None):
        """
        Initialize the structured audio segmenter.
        
        Args:
            bird_audio_dir: Directory containing class subdirectories with bird audio
            no_birds_dir: Directory containing no_birds audio files
            output_dir: Directory to save segments
            segment_duration: Duration of each segment in seconds
            sample_rate: Target sample rate
            overlap: Overlap between segments as fraction
            min_duration: Minimum duration for a segment to be saved
            max_files_per_class: Maximum number of files to process per class (for testing)
        """
        self.bird_audio_dir = Path(bird_audio_dir)
        self.no_birds_dir = Path(no_birds_dir)
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.min_duration = min_duration or (segment_duration * 0.9)
        self.max_files_per_class = max_files_per_class
        
        # Create output directories
        self.segments_dir = self.output_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        # Metadata storage
        self.segment_metadata = []
        self.ground_truth_data = []
        
        logger.info(f"Initialized structured audio segmenter")
        logger.info(f"  Bird audio dir: {self.bird_audio_dir}")
        logger.info(f"  No birds dir: {self.no_birds_dir}")
        logger.info(f"  Output dir: {self.output_dir}")
        
    def find_class_directories(self) -> Dict[str, Path]:
        """Find all class directories and the no_birds directory."""
        class_dirs = {}
        
        # Find bird class directories
        if self.bird_audio_dir.exists():
            for class_dir in self.bird_audio_dir.iterdir():
                if class_dir.is_dir() and not class_dir.name.startswith('.'):
                    class_dirs[class_dir.name] = class_dir
                    
        # Add no_birds directory
        if self.no_birds_dir.exists():
            class_dirs['no_birds'] = self.no_birds_dir
            
        logger.info(f"Found {len(class_dirs)} classes: {list(class_dirs.keys())}")
        return class_dirs
    
    def find_audio_files_in_class(self, class_dir: Path, class_name: str) -> List[Path]:
        """Find all audio files in a class directory."""
        audio_files = []
        
        for file_path in class_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        
        # Limit files if specified (for testing)
        if self.max_files_per_class:
            audio_files = audio_files[:self.max_files_per_class]
            
        logger.info(f"Found {len(audio_files)} audio files in class '{class_name}'")
        return sorted(audio_files)
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file with error handling."""
        try:
            audio, original_sr = librosa.load(file_path, sr=None, mono=True)
            logger.debug(f"Loaded {file_path.name}: {len(audio)/original_sr:.2f}s at {original_sr}Hz")
            return audio, original_sr
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None, None
    
    def resample_audio(self, audio: np.ndarray, original_sr: int) -> np.ndarray:
        """Resample audio to target sample rate if needed."""
        if original_sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.sample_rate)
            logger.debug(f"Resampled from {original_sr}Hz to {self.sample_rate}Hz")
        return audio
    
    def segment_audio_file(self, audio: np.ndarray, file_path: Path, class_name: str) -> List[dict]:
        """Segment a single audio file into fixed-duration clips."""
        segment_samples = int(self.segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))
        min_samples = int(self.min_duration * self.sample_rate)
        
        segments = []
        segment_idx = 0
        
        # Create unique file stem
        relative_path = file_path.relative_to(file_path.parent)
        file_stem = f"{class_name}_{relative_path.stem}"
        
        for start_sample in range(0, len(audio), hop_samples):
            end_sample = min(start_sample + segment_samples, len(audio))
            segment = audio[start_sample:end_sample]
            
            # Skip segments that are too short
            if len(segment) < min_samples:
                logger.debug(f"Skipping short segment: {len(segment)/self.sample_rate:.2f}s")
                continue
            
            # Pad segment if needed (only for the last segment)
            if len(segment) < segment_samples:
                padding = segment_samples - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant', constant_values=0)
                logger.debug(f"Padded segment with {padding} samples")
            
            # Create segment filename
            segment_name = f"{file_stem}_{segment_idx:03d}.wav"
            segment_path = self.segments_dir / segment_name
            
            # Save segment
            sf.write(segment_path, segment, self.sample_rate)
            
            # Store metadata
            start_time = start_sample / self.sample_rate
            end_time = end_sample / self.sample_rate
            
            segment_info = {
                'segment_name': segment_name,
                'original_file': str(file_path),
                'class_name': class_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': len(segment) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'file_path': str(segment_path.relative_to(self.output_dir))
            }
            
            segments.append(segment_info)
            
            # Add to ground truth
            self.ground_truth_data.append({
                'segment_name': segment_name,
                'label': class_name
            })
            
            segment_idx += 1
            
            # Stop if we've reached the end
            if end_sample >= len(audio):
                break
        
        logger.debug(f"Created {len(segments)} segments from {file_path.name}")
        return segments
    
    def process_class(self, class_dir: Path, class_name: str) -> List[dict]:
        """Process all audio files in a class directory."""
        logger.info(f"Processing class: {class_name}")
        
        audio_files = self.find_audio_files_in_class(class_dir, class_name)
        if not audio_files:
            logger.warning(f"No audio files found in {class_dir}")
            return []
        
        all_segments = []
        
        for file_path in tqdm(audio_files, desc=f"Processing {class_name}", leave=False):
            # Load audio
            audio, original_sr = self.load_audio(file_path)
            if audio is None:
                continue
            
            # Resample if needed
            audio = self.resample_audio(audio, original_sr)
            
            # Segment audio
            segments = self.segment_audio_file(audio, file_path, class_name)
            all_segments.extend(segments)
        
        logger.info(f"Class '{class_name}': {len(all_segments)} segments from {len(audio_files)} files")
        return all_segments
    
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process all audio files and return metadata and ground truth DataFrames."""
        logger.info("Starting structured audio segmentation...")
        
        # Find all class directories
        class_dirs = self.find_class_directories()
        if not class_dirs:
            logger.error("No class directories found!")
            return pd.DataFrame(), pd.DataFrame()
        
        all_segments = []
        
        # Process each class
        for class_name, class_dir in tqdm(class_dirs.items(), desc="Processing classes"):
            segments = self.process_class(class_dir, class_name)
            all_segments.extend(segments)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(all_segments)
        
        # Create ground truth DataFrame
        ground_truth_df = pd.DataFrame(self.ground_truth_data)
        
        # Save metadata
        metadata_path = self.output_dir / "segments_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save ground truth
        ground_truth_path = self.output_dir / "ground_truth.csv"
        ground_truth_df.to_csv(ground_truth_path, index=False)
        logger.info(f"Saved ground truth to {ground_truth_path}")
        
        # Print summary
        total_segments = len(metadata_df)
        total_duration = metadata_df['duration'].sum()
        
        logger.info(f"Segmentation complete:")
        logger.info(f"  Total segments: {total_segments}")
        logger.info(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        logger.info(f"  Classes: {len(class_dirs)}")
        
        # Per-class summary
        class_counts = ground_truth_df['label'].value_counts()
        logger.info(f"  Segments per class:")
        for class_name, count in class_counts.items():
            logger.info(f"    {class_name}: {count}")
        
        return metadata_df, ground_truth_df


def main():
    parser = argparse.ArgumentParser(description="Segment structured audio files for benchmark evaluation")
    parser.add_argument("bird_audio_dir", help="Directory containing class subdirectories with bird audio")
    parser.add_argument("no_birds_dir", help="Directory containing no_birds audio files")
    parser.add_argument("output_dir", help="Directory to save segments and metadata")
    parser.add_argument("--duration", type=float, default=3.0, 
                       help="Segment duration in seconds (default: 3.0)")
    parser.add_argument("--sample-rate", type=int, default=32000,
                       help="Target sample rate (default: 32000)")
    parser.add_argument("--overlap", type=float, default=0.0,
                       help="Overlap between segments as fraction (default: 0.0)")
    parser.add_argument("--min-duration", type=float, default=None,
                       help="Minimum segment duration (default: 90%% of segment duration)")
    parser.add_argument("--max-files-per-class", type=int, default=None,
                       help="Maximum files to process per class (for testing)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.bird_audio_dir):
        logger.error(f"Bird audio directory does not exist: {args.bird_audio_dir}")
        return 1
    
    if not os.path.exists(args.no_birds_dir):
        logger.error(f"No birds directory does not exist: {args.no_birds_dir}")
        return 1
    
    if args.overlap < 0 or args.overlap >= 1:
        logger.error("Overlap must be between 0 and 1")
        return 1
    
    # Create segmenter and process
    segmenter = StructuredAudioSegmenter(
        bird_audio_dir=args.bird_audio_dir,
        no_birds_dir=args.no_birds_dir,
        output_dir=args.output_dir,
        segment_duration=args.duration,
        sample_rate=args.sample_rate,
        overlap=args.overlap,
        min_duration=args.min_duration,
        max_files_per_class=args.max_files_per_class
    )
    
    metadata_df, ground_truth_df = segmenter.process_all()
    
    if len(metadata_df) == 0:
        logger.error("No segments were created!")
        return 1
    
    logger.info("Structured audio segmentation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 