#!/usr/bin/env python3
"""
Audio Segmentation Script for Benchmark Pipeline

Takes long audio recordings (e.g., from AudioMoth) and splits them into 
fixed-duration segments for model evaluation.

Features:
- Handles various audio formats
- Consistent naming convention
- Metadata preservation
- Quality checks for segments
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
from typing import List, Tuple, Optional
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AudioSegmenter:
    """
    Segments long audio recordings into fixed-duration clips for model evaluation.
    """
    
    def __init__(self, 
                 input_dir: str, 
                 output_dir: str, 
                 segment_duration: float = 3.0,
                 sample_rate: int = 32000,
                 overlap: float = 0.0,
                 min_duration: float = None):
        """
        Initialize the audio segmenter.
        
        Args:
            input_dir: Directory containing audio files to segment
            output_dir: Directory to save segments
            segment_duration: Duration of each segment in seconds (default: 3.0 for model consistency)
            sample_rate: Target sample rate (default: 32000 Hz)
            overlap: Overlap between segments as fraction (0.0 = no overlap, 0.5 = 50% overlap)
            min_duration: Minimum duration for a segment to be saved (default: 90% of segment_duration)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.overlap = overlap
        self.min_duration = min_duration or (segment_duration * 0.9)
        
        # Create output directories
        self.segments_dir = self.output_dir / "segments"
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported audio formats
        self.supported_formats = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        
        # Metadata storage
        self.segment_metadata = []
        
    def find_audio_files(self) -> List[Path]:
        """Find all supported audio files in input directory."""
        audio_files = []
        for file_path in self.input_dir.rglob("*"):
            if file_path.suffix.lower() in self.supported_formats:
                audio_files.append(file_path)
        
        logger.info(f"Found {len(audio_files)} audio files")
        return sorted(audio_files)
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """
        Load audio file with error handling.
        
        Returns:
            audio: Audio data
            original_sr: Original sample rate
        """
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
    
    def segment_audio(self, audio: np.ndarray, file_stem: str) -> List[dict]:
        """
        Segment audio into fixed-duration clips.
        
        Returns:
            List of segment metadata dictionaries
        """
        segment_samples = int(self.segment_duration * self.sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))
        min_samples = int(self.min_duration * self.sample_rate)
        
        segments = []
        segment_idx = 0
        
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
                'original_file': file_stem,
                'start_time': start_time,
                'end_time': end_time,
                'duration': len(segment) / self.sample_rate,
                'sample_rate': self.sample_rate,
                'file_path': str(segment_path.relative_to(self.output_dir))
            }
            
            segments.append(segment_info)
            segment_idx += 1
            
            # Stop if we've reached the end
            if end_sample >= len(audio):
                break
        
        logger.info(f"Created {len(segments)} segments from {file_stem}")
        return segments
    
    def process_file(self, file_path: Path) -> List[dict]:
        """Process a single audio file."""
        logger.info(f"Processing: {file_path.name}")
        
        # Load audio
        audio, original_sr = self.load_audio(file_path)
        if audio is None:
            return []
        
        # Resample if needed
        audio = self.resample_audio(audio, original_sr)
        
        # Create unique file stem (handle duplicates across subdirectories)
        relative_path = file_path.relative_to(self.input_dir)
        file_stem = str(relative_path.with_suffix('')).replace('/', '_').replace('\\', '_')
        
        # Segment audio
        segments = self.segment_audio(audio, file_stem)
        
        return segments
    
    def process_all(self) -> pd.DataFrame:
        """Process all audio files and return metadata DataFrame."""
        logger.info("Starting audio segmentation...")
        
        audio_files = self.find_audio_files()
        if not audio_files:
            logger.warning("No audio files found!")
            return pd.DataFrame()
        
        all_segments = []
        
        # Process each file
        for file_path in tqdm(audio_files, desc="Segmenting audio files"):
            segments = self.process_file(file_path)
            all_segments.extend(segments)
        
        # Create metadata DataFrame
        metadata_df = pd.DataFrame(all_segments)
        
        # Save metadata
        metadata_path = self.output_dir / "segments_metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Print summary
        total_segments = len(metadata_df)
        total_duration = metadata_df['duration'].sum()
        logger.info(f"Segmentation complete:")
        logger.info(f"  Total segments: {total_segments}")
        logger.info(f"  Total duration: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
        logger.info(f"  Segments saved in: {self.segments_dir}")
        
        return metadata_df


def main():
    parser = argparse.ArgumentParser(description="Segment audio files for benchmark evaluation")
    parser.add_argument("input_dir", help="Directory containing audio files to segment")
    parser.add_argument("output_dir", help="Directory to save segments and metadata")
    parser.add_argument("--duration", type=float, default=3.0, 
                       help="Segment duration in seconds (default: 3.0)")
    parser.add_argument("--sample-rate", type=int, default=32000,
                       help="Target sample rate (default: 32000)")
    parser.add_argument("--overlap", type=float, default=0.0,
                       help="Overlap between segments as fraction (default: 0.0)")
    parser.add_argument("--min-duration", type=float, default=None,
                       help="Minimum segment duration (default: 90%% of segment duration)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1
    
    if args.overlap < 0 or args.overlap >= 1:
        logger.error("Overlap must be between 0 and 1")
        return 1
    
    # Create segmenter and process
    segmenter = AudioSegmenter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        segment_duration=args.duration,
        sample_rate=args.sample_rate,
        overlap=args.overlap,
        min_duration=args.min_duration
    )
    
    metadata_df = segmenter.process_all()
    
    if len(metadata_df) == 0:
        logger.error("No segments were created!")
        return 1
    
    logger.info("Audio segmentation completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main()) 