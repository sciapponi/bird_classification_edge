#!/usr/bin/env python3
"""
Preprocessing script to apply the same preprocessing pipeline used during training.
This creates preprocessed files that can be loaded much faster than processing audio on-the-fly.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the preprocessing functions
from datasets.audio_utils import extract_call_segments, apply_bandpass_filter

def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    try:
        return file_path.stat().st_size / (1024 * 1024)
    except:
        return 0.0

def detect_audio_format(file_path: Path) -> str:
    """Detect the original audio format from file extension."""
    return file_path.suffix.lower()

def save_preprocessed_file(audio_data: np.ndarray, output_path: Path, 
                         sample_rate: int, format_type: str = 'wav') -> bool:
    """
    Save preprocessed audio data to file.
    
    Args:
        audio_data: Audio data as numpy array
        output_path: Output file path 
        sample_rate: Sample rate
        format_type: Output format ('wav', 'mp3', 'npy', 'npz', 'npz_compressed', 'preserve')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'npy':
            # Save as numpy array (fastest loading)
            np.save(output_path, audio_data.astype(np.float32))
            
        elif format_type == 'npz':
            # Save as uncompressed npz
            np.savez(output_path, audio=audio_data.astype(np.float32), sr=sample_rate)
            
        elif format_type == 'npz_compressed':
            # Save as compressed npz  
            np.savez_compressed(output_path, audio=audio_data.astype(np.float32), sr=sample_rate)
            
        else:
            # Save as audio file (wav, mp3, etc.)
            if audio_data.ndim == 1:
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
            else:
                audio_tensor = torch.from_numpy(audio_data).float()
                
            # Ensure we have the right number of channels
            if audio_tensor.shape[0] != 1:
                audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
                
            torchaudio.save(str(output_path), audio_tensor, sample_rate)
            
        return True
        
    except Exception as e:
        logger.error(f"Failed to save {output_path}: {e}")
        return False

def determine_output_path(input_path: Path, output_dir: Path, 
                         original_format: str, save_format: str) -> Path:
    """Determine the output file path based on input and format settings."""
    
    # Preserve directory structure
    relative_path = input_path.relative_to(input_path.parent.parent)
    
    if save_format == 'preserve':
        # Keep original extension
        output_path = output_dir / relative_path
    else:
        # Change extension based on format
        if save_format == 'npy':
            extension = '.npy'
        elif save_format in ['npz', 'npz_compressed']:
            extension = '.npz'
        elif save_format == 'wav':
            extension = '.wav'
        elif save_format == 'mp3':
            extension = '.mp3'
        else:
            extension = '.wav'  # fallback
            
        output_path = output_dir / relative_path.with_suffix(extension)
    
    return output_path

def process_single_file(input_path: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single audio file.
    
    Returns:
        dict: Processing results with success status and metadata
    """
    try:
        # Detect original format
        original_format = detect_audio_format(input_path)
        
        # Get file size
        original_size = get_file_size_mb(input_path)
        
        # Extract call segments using the same pipeline as training
        call_intervals, segments, original_audio, sr, duration = extract_call_segments(
            str(input_path),
            clip_duration=config['clip_duration'],
            sr=config['sample_rate'],
            lowcut=config['lowcut'],
            highcut=config['highcut']
        )
        
        if not segments or len(segments) == 0:
            # No calls detected, use fallback: process the whole file
            logger.warning(f"No calls detected in {input_path}, using fallback processing")
            
            # Load the whole file and apply same preprocessing
            try:
                waveform, sample_rate = torchaudio.load(str(input_path))
                
                # Convert to mono if stereo
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Resample if needed
                if sample_rate != config['sample_rate']:
                    resampler = torchaudio.transforms.Resample(sample_rate, config['sample_rate'])
                    waveform = resampler(waveform)
                
                # Apply bandpass filter
                audio_np = waveform.squeeze().numpy()
                filtered_audio = apply_bandpass_filter(
                    audio_np, config['sample_rate'], 
                    config['lowcut'], config['highcut']
                )
                
                # Ensure we have the right length (3 seconds)
                target_length = int(config['clip_duration'] * config['sample_rate'])
                if len(filtered_audio) > target_length:
                    # Take the first 3 seconds
                    audio_data = filtered_audio[:target_length]
                elif len(filtered_audio) < target_length:
                    # Pad with zeros
                    audio_data = np.pad(filtered_audio, (0, target_length - len(filtered_audio)))
                else:
                    audio_data = filtered_audio
                    
                fallback_used = True
                
            except Exception as e:
                logger.error(f"Fallback processing failed for {input_path}: {e}")
                return {
                    'success': False,
                    'error': str(e),
                    'original_size_mb': original_size,
                    'processed_size_mb': 0.0,
                    'calls_extracted': 0,
                    'fallback_used': False
                }
        else:
            # Use the first segment only (like training pipeline)
            audio_data = segments[0]
            fallback_used = False
            logger.info(f"Using first segment from {len(segments)} detected calls in {input_path.name}")
        
        # Determine output path
        output_path = determine_output_path(input_path, output_dir, original_format, config['save_format'])
        
        # Save preprocessed audio
        success = save_preprocessed_file(
            audio_data, output_path, config['sample_rate'], config['save_format']
        )
        
        if not success:
            return {
                'success': False,
                'error': 'Failed to save file',
                'original_size_mb': original_size,
                'processed_size_mb': 0.0,
                'calls_extracted': 1 if not fallback_used else 0,
                'fallback_used': fallback_used
            }
        
        # Get processed file size
        processed_size = get_file_size_mb(output_path)
        
        return {
            'success': True,
            'original_size_mb': original_size,
            'processed_size_mb': processed_size,
            'calls_extracted': 1 if not fallback_used else 0,
            'fallback_used': fallback_used,
            'output_path': str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")
        return {
            'success': False,
            'error': str(e),
            'original_size_mb': get_file_size_mb(input_path),
            'processed_size_mb': 0.0,
            'calls_extracted': 0,
            'fallback_used': False
        }

def load_species_list(species_file: Path) -> list:
    """Load species list from file."""
    try:
        with open(species_file, 'r', encoding='utf-8') as f:
            species = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(species)} species from {species_file}")
        for i, species_name in enumerate(species, 1):
            logger.info(f"  {i}. {species_name}")
        return species
    except Exception as e:
        logger.error(f"Failed to load species file {species_file}: {e}")
        sys.exit(1)

def find_audio_files(input_dir: Path, extensions: list = None, allowed_species: list = None) -> list:
    """Find all audio files in directory and subdirectories, optionally filtered by species."""
    if extensions is None:
        extensions = ['.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg']
    
    audio_files = []
    
    if allowed_species is not None:
        # Filter by specific species directories
        logger.info(f"Filtering files by {len(allowed_species)} specified species:")
        for species in allowed_species:
            species_dir = input_dir / species
            if species_dir.exists() and species_dir.is_dir():
                logger.info(f"  Processing species: {species}")
                species_files = []
                for ext in extensions:
                    species_files.extend(species_dir.glob(f'*{ext}'))
                    species_files.extend(species_dir.glob(f'*{ext.upper()}'))
                audio_files.extend(species_files)
                logger.info(f"    Found {len(species_files)} files in {species}")
            else:
                logger.warning(f"  Species directory not found: {species_dir}")
    else:
        # Process all files (original behavior)
        for ext in extensions:
            audio_files.extend(input_dir.rglob(f'*{ext}'))
            audio_files.extend(input_dir.rglob(f'*{ext.upper()}'))
    
    return sorted(audio_files)

def process_all_files(input_dir: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    """Process all audio files in the input directory."""
    
    # Find all audio files (filtered by species if specified)
    allowed_species = config.get('allowed_species', None)
    audio_files = find_audio_files(input_dir, allowed_species=allowed_species)
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'calls_extracted': 0,
            'fallback_used': 0,
            'original_size_mb': 0.0,
            'processed_size_mb': 0.0
        }
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process files
    results = {
        'total_files': len(audio_files),
        'processed_files': 0,
        'failed_files': 0,
        'calls_extracted': 0,
        'fallback_used': 0,
        'original_size_mb': 0.0,
        'processed_size_mb': 0.0
    }
    
    # Create progress bar
    pbar = tqdm(audio_files, desc="Processing audio files")
    
    for file_path in pbar:
        # Update progress bar description
        pbar.set_description(f"Processing {file_path.name}")
        
        # Process the file
        result = process_single_file(file_path, output_dir, config)
        
        # Update statistics
        results['original_size_mb'] += result['original_size_mb']
        results['processed_size_mb'] += result['processed_size_mb']
        
        if result['success']:
            results['processed_files'] += 1
            results['calls_extracted'] += result['calls_extracted']
            if result['fallback_used']:
                results['fallback_used'] += 1
        else:
            results['failed_files'] += 1
            logger.error(f"Failed to process {file_path}: {result.get('error', 'Unknown error')}")
        
        # Update progress bar postfix
        pbar.set_postfix({
            'processed': results['processed_files'],
            'failed': results['failed_files'],
            'calls': results['calls_extracted']
        })
    
    return results

def save_final_report(output_dir: Path, config: Dict[str, Any], statistics: Dict[str, Any], 
                     processing_time: float, input_dir: Path):
    """Save a comprehensive final processing report."""
    
    report = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3],
        'config': config,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'statistics': {
            **statistics,
            'processing_time': processing_time
        }
    }
    
    report_path = output_dir / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Final processing report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess audio files using training pipeline')
    parser.add_argument('--input-dir', type=str, default='bird_sound_dataset', 
                       help='Input directory containing audio files (default: bird_sound_dataset)')
    parser.add_argument('--output-dir', type=str, default='bird_sound_dataset_preprocessed', 
                       help='Output directory for preprocessed files (default: bird_sound_dataset_preprocessed)')
    parser.add_argument('--sample-rate', type=int, default=32000, help='Target sample rate (default: 32000)')
    parser.add_argument('--clip-duration', type=float, default=3.0, help='Clip duration in seconds (default: 3.0)')
    parser.add_argument('--lowcut', type=float, default=150.0, help='Bandpass filter low cutoff (default: 150.0)')
    parser.add_argument('--highcut', type=float, default=16000.0, help='Bandpass filter high cutoff (default: 16000.0)')
    parser.add_argument('--format', type=str, choices=['wav', 'mp3', 'npy', 'npz', 'npz_compressed', 'preserve'], 
                       default='preserve', help='Output format (default: preserve)')
    parser.add_argument('--species-file', type=str, default=None,
                       help='File containing list of species to process (one per line). If not provided, processes all species.')
    
    args = parser.parse_args()
    
    # Convert paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load species list if provided
    allowed_species = None
    if args.species_file:
        species_file_path = Path(args.species_file)
        if not species_file_path.exists():
            logger.error(f"Species file does not exist: {species_file_path}")
            sys.exit(1)
        allowed_species = load_species_list(species_file_path)
    
    # Configuration
    config = {
        'sample_rate': args.sample_rate,
        'clip_duration': args.clip_duration,
        'lowcut': args.lowcut,
        'highcut': args.highcut,
        'extract_calls': True,
        'save_format': args.format,
        'allowed_species': allowed_species
    }
    
    logger.info("Starting preprocessing with configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Process all files
    start_time = time.time()
    statistics = process_all_files(input_dir, output_dir, config)
    processing_time = time.time() - start_time
    
    # Save final report
    save_final_report(output_dir, config, statistics, processing_time, input_dir)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PREPROCESSING COMPLETED")
    logger.info("="*50)
    logger.info(f"Total files found: {statistics['total_files']}")
    logger.info(f"Successfully processed: {statistics['processed_files']}")
    logger.info(f"Failed: {statistics['failed_files']}")
    logger.info(f"Calls extracted: {statistics['calls_extracted']}")
    logger.info(f"Fallback used: {statistics['fallback_used']}")
    logger.info(f"Original size: {statistics['original_size_mb']:.2f} MB")
    logger.info(f"Processed size: {statistics['processed_size_mb']:.2f} MB")
    logger.info(f"Compression ratio: {statistics['processed_size_mb']/statistics['original_size_mb']:.2f}x")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main() 